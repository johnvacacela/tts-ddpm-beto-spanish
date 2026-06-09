""" from https://github.com/keithito/tacotron """
'''
Defines the set of symbols used in text input to the model.
'''
from .cmudict import valid_symbols
_arpabet = ['@' + s for s in valid_symbols]

def get_symbols(symbol_set='english_basic'):
    if symbol_set == 'english_basic':
        _pad = '_'
        _punctuation = '!\'(),.:;? '
        _special = '-'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == 'english_basic_lowercase':
        _pad = '_'
        _punctuation = '!\'"(),.:;? '
        _special = '-'
        _letters = 'abcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == 'english_expanded':
        _punctuation = '!\'",.:;? '
        _math = '#%&*+-/[]()'
        _special = '_@©°½—₩€$'
        _accented = 'áçéêëñöøćž'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        symbols = list(_punctuation + _math + _special + _accented + _letters) + _arpabet
    elif symbol_set == 'ipa_es':
        _pad     = ['_']
        _special = ['-']
        _punct   = list('!(),.:;? ')
        _ipa     = list('abdefijklmnoprstuwx')
        _ipa_ext = ['ð','ŋ','ɛ','ɡ','ɣ','ɪ','ɲ','ɾ','ʃ','ʊ','ʎ','ʒ','ʝ','ˈ','ˌ','β','θ']
        # Total: 1 padding + 1 separador + 9 puntuacion +
        # 19 IPA base + 17 IPA Unicode = 47 simbolos.
        symbols  = _pad + _special + _punct + _ipa + _ipa_ext
    else:
        raise Exception("{} symbol set does not exist".format(symbol_set))
    return symbols

def get_pad_idx(symbol_set='english_basic'):
    if symbol_set in {'english_basic', 'english_basic_lowercase', 'ipa_es'}:
        return 0
    else:
        raise Exception("{} symbol set not used yet".format(symbol_set))
