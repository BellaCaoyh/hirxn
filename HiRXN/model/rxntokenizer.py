import logging
from rdkit import Chem
import collections
import re
logger = logging.getLogger(__name__)

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def whitespace_tokenize_ac(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = re.split(r'[.*]', text)
    # tokens = re.findall(r'[^.*]+|[.*]', text)
    # tokens = text.split('*')
    return tokens

def tokenizer():
    pass

def tokenize_decompse(text,r):#通过提取化合物中基团来判断vocab
    # vocab = load_vocab('../data/vocab.txt')
    unk_token = '[UNK]'
    output_tokens = []
    cur_substr = None
    sub_tokens = []
    # if text == '>>' or text =='.' or text == '*':
    #     output_tokens.extend(text)
    #     return output_tokens
    if text == '>>':
        output_tokens.extend(text)
        return output_tokens

    '''
    r= 0r=1 and r=2 for uspto
    '''
    # for i in range(0,r+1):
    #     if i == 0:
    #         pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    #         regex = re.compile(pattern)
    #         tokens = [token for token in regex.findall(text)]
    #         output_tokens.extend(tokens)
    #     else:
    #         decompse_vector = decompse_ac(text,i)
    #         for key,value in decompse_vector.items():
    #                 cur_substr = key
    #                 sub_tokens.append(cur_substr)
    #         if cur_substr == None:
    #             output_tokens.append(unk_token)
    #         else:
    #             output_tokens.extend(sub_tokens)
    '''
    for suzuki BH denmark        
    '''
    for i in range(0,r+1):
        sub_tokens=[]
        decompse_vector = decompse_ac(text,i)
        for key,value in decompse_vector.items():
                cur_substr = key
                sub_tokens.append(cur_substr)
        if cur_substr == None:
            output_tokens.append(unk_token)
        else:
            output_tokens.extend(sub_tokens)
    '''
    for experiment
    '''
    # sub_tokens=[]
    # decompse_vector = decompse_ac(text,r)
    # for key,value in decompse_vector.items():
    #         cur_substr = key
    #         sub_tokens.append(cur_substr)
    # if cur_substr == None:
    #     output_tokens.append(unk_token)
    # else:
    #     output_tokens.extend(sub_tokens)
    return output_tokens

def tokenize_ac(text,r):
    split_tokens = []
    output_tokens = spilt_from_middle(text)
    for token in output_tokens:
        for sub_token in tokenize_decompse(token,r):
            split_tokens.append(sub_token)
    return split_tokens

def spilt_from_middle(text):

    # text = self._tokenize_chinese_chars(text)
    orig_tokens = whitespace_tokenize_ac(text)
    split_tokens = []
    for token in orig_tokens:
        split_tokens.extend(run_split_on_punc(token))
    output_tokens = split_tokens
    return output_tokens

def run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation_ac(char):
            if chars[i-1] != char:#提取出完整的“>>”
                output.append([char])
            else:
                output[-1].append(char)
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1
    return ["".join(x) for x in output]

def _is_punctuation_ac(char):
    """Checks whether `chars` is a punctuation character。修改了该部分内容，仅检测是否是'.'或者是'>>'"""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if cp == 62:#46 62分别是.和>的unicode码点
        return True
    return False

def convert_tokens_to_ids(tokens,len_limit):
    """Converts a sequence of tokens into ids using the vocab."""
    vocab = load_vocab('../data/vocab.txt')
    max_len = len_limit
    ids = []
    for token in tokens:
        ids.append(vocab[token])
    if len(ids) > max_len:
        logger.warning(
            "Token indices sequence length is longer than the specified maximum "
            " sequence length for this BERT model ({} > {}). Running this"
            " sequence through BERT will result in indexing errors".format(len(ids), max_len)
        )
    return ids

def count_substructures(radius,molecule):
    """Helper function for get the information of molecular signature of a
    metabolite. The relaxed signature requires the number of each substructure
    to construct a matrix for each molecule.
    Parameters
    ----------
    radius : int
        the radius is bond-distance that defines how many neighbor atoms should
        be considered in a reaction center.
    molecule : Molecule
        a molecule object create by RDkit (e.g. Chem.MolFromInchi(inchi_code)
        or Chem.MolToSmiles(smiles_code))
    Returns
    -------
    dict
        dictionary of molecular signature for a molecule,
        {smiles: molecular_signature}
    """
    m = molecule
    smi_count = dict()
    atomList = [atom for atom in m.GetAtoms()]

    for i in range(len(atomList)):
        env = Chem.FindAtomEnvironmentOfRadiusN(m,radius,i)
        atoms=set()
        for bidx in env:
            atoms.add(m.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.add(m.GetBondWithIdx(bidx).GetEndAtomIdx())

        # only one atom is in this environment, such as O in H2O
        if len(atoms) == 0:
            atoms = {i}

        smi = Chem.MolFragmentToSmiles(m,atomsToUse=list(atoms),
                                    bondsToUse=env,canonical=True)

        if smi in smi_count:
            smi_count[smi] = smi_count[smi] + 1
        else:
            smi_count[smi] = 1
    return smi_count
def decompse_ac(db_smiles,radius=1):#提取一个化合中的基团
    non_decomposable = []
    decompose_vector = None

    smiles_pH7 = db_smiles
    try:
        mol = Chem.MolFromSmiles(smiles_pH7)
        mol = Chem.RemoveHs(mol)
        # Chem.RemoveStereochemistry(mol)
        smi_count = count_substructures(radius,mol)
        decompose_vector = smi_count

    except Exception as e:
        non_decomposable.append(0)

    return decompose_vector

def get_word2id(self, datas):
    word_freq = {}
    for data in datas:
        for word in data:
            word_freq[word] = word_freq.get(word, 0) + 1
    word2id = {"<pad>": 0, "<unk>": 1}
    for word in word_freq:
        if word_freq[word] < self.max_count:
            continue
        else:
            word2id[word] = len(word2id)
    return word2id

def rxntokenizer(content,r=1):
    token = tokenize_ac(content,r)
    # token_ids = config.tokenizer.convert_tokens_to_ids(token)

    '''去除重复基团'''
    # a = token.index('>')
    # substrate = token[:a]
    # product = token[a + 2:]
    # n_substrate = substrate.copy()
    # for i in substrate:
    #     if i in product:
    #         product.remove(i)
    #         n_substrate.remove(i)
    # token = n_substrate + ['>', '>'] + product

    # token =token
    # seq_len = len(token)
    # mask = []
    # token_ids = convert_tokens_to_ids(token,len_limit)
    return token


if __name__=='__main__':
    rxn_smiles= "O=C(/N=C/c1ccccc1)c1ccccc1.CCS.O=P1(O)Oc2c(Br)cc3ccccc3c2-c2c(c(Br)cc3ccccc23)O1>>CCSC(NC(=O)c1ccccc1)c1ccccc1"
    res = rxntokenizer(rxn_smiles, r=2)
    print(len(res))
    # tokens = {}
    # tokens_non = {}
    # for i in range(3):
    #     token = rxntokenizer(rxn_smiles, r=i)
    #     tokens_non[i] = token
    #     if i>0:
    #         tokens[i] = list(set(token)-set(tokens_non[i-1]))
    #     else:
    #         tokens[i] = token