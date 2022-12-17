import pdb

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForMaskedLM, AdamW, BertConfig, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaForMaskedLM, RobertaConfig
import random
from copy import copy
import regex as re
import pandas as pd
from tqdm import tqdm,trange
import argparse
import logging

from train_util import *
from infer_util import get_gendered_profs

from model import *

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        logger.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
    else:
        logger.info('No GPU available, using the CPU instead.')

def data_formatter_inherent(lines, lines_anti, filename, mask_token='[MASK]', baseline_tester=False, reverse=True, female_names=['woman'], male_names=['man']):

    # Initialise
    masked_data = []
    masklabels = []
    professions = []

    if baseline_tester:
        mprofs, fprofs = get_gendered_profs()

    textfile = open(filename + '.txt', 'w')

    for i, line in enumerate(lines):
        female_name = random.choice(female_names)
        male_name = random.choice(male_names)
        mask_regex = r"(\[he\]|\[she\]|\[him\]|\[his\]|\[her\]|\[He\]|\[She\]|\[His\]|\[Her\])"
        pronoun = re.findall(mask_regex, line)

        if len(pronoun) == 1: 
            pronoun = pronoun[0][1:-1]
            pronoun_anti = re.findall(mask_regex, lines_anti[i])[0][1:-1]

            # Remove number at start of line
            new_line = re.sub(r"^(\d*)", "", line)
            new_line = re.sub(r"(.)$", " . ", new_line[1:])

            profession_pre = re.findall('\[(.*?)\]', new_line)[0]

            if profession_pre[1:4] == 'he ':
                profession = profession_pre[4:]  # i.e. the/The

            elif profession_pre[0:2] == 'a ':
                profession = profession_pre[2:]

            else:
                profession = profession_pre

            professions.append(profession) 

            new_line = re.sub(mask_regex, mask_token, new_line) 

            new_line = re.sub(r'\[(.*?)\]', lambda L: L.group(1).rsplit('|', 1)[-1], new_line)

            # replace square brackets on MASK
            new_line = re.sub('MASK', '[MASK]', new_line)

            if reverse:
                new_line_rev = copy(new_line)

            if baseline_tester:
                if pronoun in ('she', 'her'):
                    new_line = new_line.replace(profession_pre, female_name)

                else:
                    new_line = new_line.replace(profession_pre, male_name)
                if baseline_tester == 1:
                    for prof in mprofs:
                        new_line = new_line.replace('The ' + prof, male_name)
                        new_line = new_line.replace('the ' + prof, male_name)
                        new_line = new_line.replace('a ' + prof, male_name)
                        new_line = new_line.replace('A ' + prof, male_name)

                    for prof in fprofs:
                        new_line = new_line.replace('The ' + prof, female_name)
                        new_line = new_line.replace('the ' + prof, female_name)
                        new_line = new_line.replace('a ' + prof, female_name)
                        new_line = new_line.replace('A ' + prof, female_name)

            new_line = new_line.lstrip().rstrip()
            masked_data.append(new_line)
            textfile.write(new_line + '\n')
            masklabels.append([pronoun, pronoun_anti])

            if reverse and baseline_tester:
                if pronoun in ('she', 'her'):
                    new_line_rev = new_line_rev.replace(profession_pre, male_name)

                else:
                    new_line_rev = new_line_rev.replace(profession_pre, female_name)

                if baseline_tester == 2:
                    for prof in fprofs:
                        new_line_rev = new_line_rev.replace('The ' + prof, male_name)
                        new_line_rev = new_line_rev.replace('the ' + prof, male_name)
                        new_line_rev = new_line_rev.replace('a ' + prof, male_name)
                        new_line_rev = new_line_rev.replace('A ' + prof, male_name)

                    for prof in mprofs:
                        new_line_rev = new_line_rev.replace('The ' + prof, female_name)
                        new_line_rev = new_line_rev.replace('the ' + prof, female_name)
                        new_line_rev = new_line_rev.replace('a ' + prof, female_name)
                        new_line_rev = new_line_rev.replace('A ' + prof, female_name)

                textfile.write(new_line_rev)
                masked_data.append(new_line_rev)
                masklabels.append([pronoun_anti, pronoun])
                professions.append('removed prof')

    textfile.close()


    return masked_data, masklabels, professions


# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
def save_model(processed_model, epoch, tokenizer, args):
    lr, eps = args.learning_rate, args.adam_epsilon
    output_dir = './model_save/{}_{}/epoch_{}/'.format(args.save_path, args.data, epoch)

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = processed_model.module if hasattr(processed_model, 'module') else processed_model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save([epoch, lr, eps], os.path.join(output_dir, 'training_args.bin'))


def train(data, args):

    sentences = data.text.values
    labels = data.pronouns.values
    
    # Report the number of sentences.
    print('Number of training sentences: {:,}\n'.format(data.shape[0]))

    print('Loading BERT-debias ...')
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.output_hidden_states = True
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # neutral tokenization
    if args.stereo_only == True:
        neutral_list = load_stereo("./data/stereotype_list.tsv")
    else:
        neutral_list = load_file("./data/no_gender_list.tsv")
        neutral_list += load_stereo("./data/stereotype_list.tsv")

    neutral_tok = tokenizing_neutral(neutral_list, tokenizer)

    # gender pair list
    female_list = load_file("./data/female_word_file.txt")
    male_list = load_file("./data/male_word_file.txt")
    gender_pairs = {"male": male_list, "female": female_list}

    # build tensor dataset
    features = convert_examples_to_features(tokenizer, sentences, labels, neutral_tok, args)
    dataset = convert_features_to_dataset(features)

    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    with open("corefBias/WinoBias/wino/data/pro_stereotyped_type1.txt.test") as f1:
         pro_test2 = f1.readlines()
    with open("corefBias/WinoBias/wino/data/anti_stereotyped_type1.txt.test") as f2:
         anti_test2 = f2.readlines()

    base_masked_data, base_labels, base_professions = data_formatter_inherent(pro_test2, anti_test2, "test2_formatted", baseline_tester=1)

    model = BERT_debias("bert-base-uncased", config, args, train_dataloader, base_masked_data, base_labels, tokenizer)

    model.cuda()

    set_seed(args)

    optimizer = AdamW(model.parameters(),
                    lr = args.learning_rate, 
                    eps = args.adam_epsilon 
                    )

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * args.num_train_epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # Store the average loss after each epoch so we can plot them.
    training_loss_values = []
    eval_loss_values = []

    epoch_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    # calculate gender-directional vector every epoch
    gender_vector = calculate_gender_vector(gender_pairs, tokenizer, model)
    normed_gender_vector = gender_vector / torch.norm(gender_vector, p=2)
    
    # For each epoch...
    for epoch_i in epoch_iterator:
#         print("epoch {}".format(str(epoch_i+1)))
        t0 = time.time()
        total_loss = 0
        total_mlm_loss = 0
        total_orthogonal_loss = 0
        total_ewc_loss = 0

        batch_iterator = tqdm(train_dataloader, desc="Train Iteration")

        # For each batch of training data...
        model.train()
        for step, batch in enumerate(batch_iterator):

            b_input_ids = batch[0].to(args.device)
            b_input_mask = batch[1].to(args.device)
            b_labels = batch[2].to(args.device)
            b_debias_labels = batch[3].to(args.device)

            model.zero_grad()

            mlm_loss, orthogonal_loss = model(input_ids=b_input_ids, attention_mask=b_input_mask,
                        labels=b_labels, debias_label_ids=b_debias_labels,
                        gender_vector=normed_gender_vector.detach())

            importance = args.ewc_imp

            loss = mlm_loss + args.lambda_loss * orthogonal_loss
            loss = loss + importance * model.penalty()

            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_orthogonal_loss += args.lambda_loss * orthogonal_loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        avg_mlm_loss = total_mlm_loss / len(train_dataloader)
        avg_orth_loss = total_orthogonal_loss / len(train_dataloader)

        training_loss_values.append(avg_train_loss)
        save_model(model, epoch_i+1, tokenizer, args)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Average mlm loss: {0:.2f}".format(avg_mlm_loss))
        print("  Average orthogonal loss: {0:.2f}".format(avg_orth_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
        
        model.train(False)

    print("")
    print("Training complete!")
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="Random seed for initialization")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_validation", action="store_true", help="Whether to run validation")
    parser.add_argument("--model_name", default="bert_debias", type=str, help="Pre-trained model name")
    parser.add_argument("--data", default="augmented", type=str, help="Augmented (augmented) / Unaugmented Setting (unaugmented)")
    parser.add_argument("--max_seq_length", default=164, type=int, help="The maximum total input sequence length after tokenization. Sequences longer \
                than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--train_batch_size", default=6, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=50, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--warmup_step", default=0, type=int, help="step of linear warmup")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer")
    parser.add_argument("--lambda_loss", default=1.0, type=float, help="Lambda for scaling loss")
    parser.add_argument("--ewc_imp", default=0.5, type=float, help="Importance term for EWC")
    parser.add_argument("--orth_loss_ver", default="abs", type=str, help="absolute or squared")
    parser.add_argument("--stereo_only", action="store_false", help="Neutral stereo only")
    parser.add_argument("--save_path", default="./bert_debias", type=str, help="Path for saving results")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    if args.do_train:
        data = load_data(mode = args.data)
        train(data, args)
        print("end")
    

if __name__ == "__main__":
    main()