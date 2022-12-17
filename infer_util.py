from copy import copy
import numpy as np
import pandas as pd
import regex as re
import random
import glob

def generalise_profession_embeddings(string, tokenizer):
    """
    Replace true profession in string with "[profession]".

    :param str string: Input string from Winobias
    :return generalised_string: string with "[profession]"
    subbed in place of actuall profession
    :return profession: entity profession
    """
    regex_extracting_profession = r"[\s\w]*(\[[\w\s]*\])[\w\s]*(\[his\]|\[her\]|\[he\]|\[she\]|)"

    # Extract profession/gender instances in string
    profession, gender = re.findall(regex_extracting_profession, string)[0]
    # print(profession, gender)
    # print("Profession: {}, Gender: {}".format(profession, gender)) # For debugging

    # Remove brackets from
    prof_amended = profession[1:-1]
    # print(prof_amended)

    # Check if profession is multi-worded
    prof_split = prof_amended.split()

    if len(prof_split) > 1:
        # If so, replace context with multiple 'profession' templates
        prof_template = '[' + ' '.join(len(prof_split) * ['profession']) + ']'
    else:
        prof_template = "[profession]"

    generalised_string = string.replace(profession, prof_template)

    # Check if original profession is tokenised by > 1 token
    gen_tokens = tokenizer.encode(generalised_string)
    original_tokens = tokenizer.encode(string)

    # If so count the number of
    if len(original_tokens) > len(gen_tokens):
        # Find number of elements in orig string not in gen string
        diff_elems = set(original_tokens) - set(gen_tokens)
        num_elems = len(diff_elems)
        generalised_string = string.replace(
            profession, '[' + ' '.join(num_elems * ['mask']) + ']')

    return generalised_string, profession


def remove_the_from_brackets(string):
    """
    Searches for whether there is a "The" in the profession-related
    square brackets. If so, it extracts "The" and keeps only the professions
    within the brackets.

    e.g. "[The engineer] was upset..." => "The [engineer] was upset..."
    :return str string: input string with "The/the" removed from the target entity
    """
    # Idenitify whether the professional term starts with "[The ...]"
    regex = "[\s\w]*(\[The [\w\s]*\])[\w\s]*"
    profession_instance_The = re.findall(regex, string)

    # If so, pull "The" outside of the square brackets
    if len(profession_instance_The) > 0:
        replacement = "The [" + profession_instance_The[0][5:]
        string = string.replace(profession_instance_The[0], replacement)

    # Do the same for [the ...]
    # Idenitify whether the professional term starts with "[The ...]"
    regex = "[\s\w]*(\[the [\w\s]*\])[\w\s]*"
    profession_instance_the = re.findall(regex, string)

    # If so, pull "The" outside of the square brackets
    if len(profession_instance_the) > 0:
        replacement = "the [" + profession_instance_the[0][5:]
        string = string.replace(profession_instance_the[0], replacement)

    return string


def generalise_profession(string):
    """
    Replace true profession in string with "[profession]".

    :param str string: Input string from Winobias
    :return generalised_string: string with "[profession]"
    subbed in place of actuall profession
    :return profession: entity profession

    """
    regex_extracting_profession = r"[\s\w]*(\[[\w\s]*\])[\w\s]*(\[his\]|\[her\]|\[he\]|\[she\]|)"

    # Extract profession/gender instances in string
    profession, gender = re.findall(regex_extracting_profession, string)[0]
    # print(profession, gender)
    # print("Profession: {}, Gender: {}".format(profession, gender)) # For debugging

    # Test gender to check we have extracted the right quantities
    assert gender in set(["[his]", "[her]", "[he]", "[she]", "[him]"])  # For debugging (always leave on)

    # Remove brackets from
    prof_amended = profession[1:-1]
    # print(prof_amended)

    # Check if profession is multi-worded
    prof_split = prof_amended.split()

    if len(prof_split) > 1:
        # If so, replace context with multiple 'profession' templates
        prof_template = '[' + ' '.join(len(prof_split) * ['profession']) + ']'
    else:
        prof_template = "[profession]"

    generalised_string = string.replace(profession, prof_template)

    # Check if original profession is tokenised by > 1 token
    gen_tokens = tokenizer.encode(generalised_string)
    original_tokens = tokenizer.encode(string)

    # If so count the number of
    if len(original_tokens) > len(gen_tokens):
        # Find number of elements in orig string not in gen string
        diff_elems = set(original_tokens) - set(gen_tokens)
        num_elems = len(diff_elems)
        generalised_string = string.replace(
            profession, '[' + ' '.join(num_elems * ['profession']) + ']')

    return generalised_string, profession


def identify_profession_token(string, general_string):
    """
    Returns the index of the token corresponding to the string's profession
    for a particular tokenizer.
    """
    # print(string)
    # Get tokens of the raw string and the generalised string
    # return [len(string.split(']')[0])]
    orig_tokens = np.array(tokenizer.encode(string))
    gen_tokens = np.array(tokenizer.encode(general_string))

    # By comparing the difference, identify which tokens correspond to the
    # original profession
    # print(orig_tokens, gen_tokens)
    token_diff = orig_tokens - gen_tokens
    non_zero_index = np.nonzero(token_diff)[0]

    return non_zero_index.tolist()


def change_gender(string, gender):
    """
    Change string's pronoun to that corresponding to a user given gender
    """
    term_a = r'(\[his\])|(\[her\])'
    term_b = r'(\[he\])|(\[she\])'
    term_c = r'(\[him\])|(\[her\])'

    if gender == "M":
        string = re.sub(term_a, '[his]', string)
        string = re.sub(term_b, '[he]', string)
        # string = re.sub(term_c, '[him]', string)

        return string

    elif gender == 'F':
        string = re.sub(term_a, '[her]', string)
        string = re.sub(term_b, '[she]', string)
        string = re.sub(term_c, '[her]', string)

        return string
    else:
        return ValueError("Need to specify appropirate gender: 'M' or 'F'")


def extract_professional_layer(string, ind, model, tokenizer):
    """
    * Format string to remove brackets around gender/profession
    * Tokenize/Encode and find embedding representation in BERT

    return: a tuple of embeddings indexed by layer number (i.e. layers[-1] will
    be the final layer and layers[0] will be the first layer)

    Method inspired from
    https://github.com/huggingface/transformers/issues/1950
    """
    regex_extracting_profession = r"[\s\w]*(\[\w*\])[\w\s]*(\[his\]|\[her\]|\[he\]|\[she\]|)"
    profession, gender = re.findall(regex_extracting_profession, string)[0]

    # Remove brackets around profession/gender
    string = string.replace(profession, profession[1:-1])
    string = string.replace(gender, gender[1:-1])
    # print("Modified String {}".format(string))
    # print(string)
    # print(type(string))

    # Tokenize string and convert to torch.tensor
    tokens = torch.tensor(tokenizer.encode(string)).unsqueeze(0)

    # Extract embeddings by passing tokens into model and selecting 3rd return object
    # print(tokens)
    with torch.no_grad():
        outputs = model(tokens)
        outputs = outputs[2]

    assert tokens.shape[1] == outputs[0].shape[1]  # Check each token has its own embedding

    # Extract embedding from space and return as a tuple (ordered from first to last).
    number_of_layers = len(outputs)

    if len(ind) == 1:
        layers = tuple(outputs[i][0][ind][0] for i in range(13))

    # If multiple tokens for a mapping exist, take the mean
    elif len(ind) > 1:
        layers = tuple(outputs[i][0][ind][0].mean(1) for i in range(13))

    return layers


def extract_gendered_profession_emb(string, model, tokenizer):
    """
    Create template string replacing profession with a template value

    * extract profession from text
    * duplicate it ans sub with "profession" term
    * tokenise and identify which layer will relate to contextualised layer for that profession

    Returns embedding representation for a profession within a string for
    male and female pronouns. The index corresponding to the professional
    token, and the profession string itself, are also returned

    """
    string = remove_the_from_brackets(string)
    # print(string) # for debugging
    general_string, profession = generalise_profession(string)
    token_index = identify_profession_token(string, general_string)
    # if len(token_index) > 1: # Warns when more than one token is used for a profession
    #  print("""
    #    WARNING: profession for {} is represented with more than one token ({})
    #  """.format(string, token_index))
    male_string = change_gender(string, gender='M')
    female_string = change_gender(string, gender='F')

    male_representation = extract_professional_layer(
        male_string, token_index, model, tokenizer)

    female_representation = extract_professional_layer(
        female_string, token_index, model, tokenizer)

    return male_representation, female_representation, token_index, profession


def extract_full_layer(string, ind, model, tokenizer):
    """
    * Format string to remove brackets around gender/profession
    * Tokenize/Encode and find embedding representation in BERT

    return: a tuple of embeddings indexed by layer number (i.e. layers[-1] will
    be the final layer and layers[0] will be the first layer)

    Method inspired from
    https://github.com/huggingface/transformers/issues/1950
    """
    regex_extracting_profession = r"[\s\w]*(\[\w*\])[\w\s]*(\[his\]|\[her\]|\[he\]|\[she\]|)"
    profession, gender = re.findall(regex_extracting_profession, string)[0]

    # Remove brackets around profession/gender
    string = string.replace(profession, profession[1:-1])
    string = string.replace(gender, gender[1:-1])
    # print("Modified String {}".format(string))
    # print(string)
    # print(type(string))

    # Tokenize string and convert to torch.tensor
    tokens = torch.tensor(tokenizer.encode(string)).unsqueeze(0)

    # Extract embeddings by passing tokens into model and selecting 3rd return object
    # print(tokens)
    with torch.no_grad():
        outputs = model(tokens)
        outputs = outputs[2]
    assert tokens.shape[1] == outputs[0].shape[1]  # Check each token has its own embedding

    # Extract embedding from space and return as a tuple (ordered from first to last).
    number_of_layers = len(outputs)
    if len(ind) == 1:
        layers = tuple(outputs[i][0][:][0] for i in range(13))

    # If multiple tokens for a mapping exist, take the mean
    elif len(ind) > 1:
        layers = tuple(outputs[i][0][:][0].mean(1) for i in range(13))

    return layers

def get_gendered_profs():
    """
    Returns lists of stereotypically male and female professions [US Labor Statistics 2017]
    """
    # Labor statistics from US 2017 population survey
    dic_of_profs = {'carpenter': 2,'mechanic':4,'construction worker':4, 'laborer':4, 'driver':6,'sheriff':14,'mover':18, 'developer':20, 'farmer':22,'guard':22,
              'chief':27,'janitor':34,'lawyer':35,'cook':38,'physician':38,'CEO':39, 'analyst':41,'manager':43, 'supervisor':44, 'salesperson':48, 'editor':52, 'designer':54,'accountant':61,'auditor':61, 'writer':63,'baker':65,'clerk':72,
              'cashier':73, 'counselor':73, 'attendant':76, 'teacher':78, 'sewer':80, 'librarian':84, 'assistant':85, 'cleaner':89, 'housekeeper':89,'nurse':90,'receptionist':90, 'hairdresser':92, 'secretary':95}
    mprofs = []
    fprofs = []
    for key in dic_of_profs.keys():
        if dic_of_profs[key] >50:
            fprofs.append(key)
        else:
            mprofs.append(key)

    # WinoBias includes profession "tailor" that is stereotypically male [Zhao et al 2019]
    mprofs.append('tailor')

    return mprofs,fprofs
