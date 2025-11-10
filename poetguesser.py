import requests
import json
import re
import os
from importlib.resources import files
import unicodedata
from collections import Counter
import pandas as pd
from sklearn.svm import SVC


class PoetGuesser:
    '''
    Authorship recognition of Czech poetic texts based on a mixed feature set
    (versification features + delexicalized linguistic features).
    ---
    Methods:

        analyze_doc(doc:str|list)
            : Annotate input document and process it with Ingram and UDPipe

        doc_summary(verbose:bool=True)            
            : Either print summary on processed input document or return it as a dict   

        available_candidates(meter:str, sample_size:int, verbose:bool=True):
            : Either print number of available candidates and their respective samples for
            : selected meter and sample size or return it as a dict

        guess_author(
            sample_size:int, meter:str, v_ngram:int|None=3, w_ngram:int|None=1, candidate_set:list=[],
            level_authors:bool=False, random_seed:int|None=None, zscores:bool=True, custom_classifier=None
        )        
            : Predict author of input document with Support Vector Machine or any custom classifier
    '''


    def __init__(
        self, 
        base_url_ingram:str = 'https://versologie.cz/v2/tool_ingram/api.php',
        base_url_udpipe:str = 'http://lindat.mff.cuni.cz/services/udpipe/api/process',
        min_score:float     = 0.7,
    ):
        '''
        Initialize PoetGuesser class
        ---
        Params:
            base_url_ingram (str)   : base url of Ingram API (to retrieve poetic meter and rhythmical bitstrings)
            base_url_udpipe (str)   : base url of UDPipe API (to retrieve poetic meter and rhythmical bitstrings)
            min_score       (float) : minimum score for meter to be considered the meter of the input document(s)
        Returns:
            None
        '''
        self.base_url_ingram_ = base_url_ingram
        self.base_url_udpipe_ = base_url_udpipe
        self.min_score_       = min_score
        self.dataset_dir_     = files(__package__) / 'data'
        with open(files(__package__) / 'samplelist.json') as f:
            self.samplelist_ = json.load(f)


    # ***********************************************************************************************************
    # ANALYZE DOCUMENT(S)
    # ***********************************************************************************************************


    def analyze_doc(self, doc:str|list):
        '''
        Annotate input document and process it with Ingram and UDPipe
        --- 
        Params:
            doc (str|list) : Input document(s) passed either as a string or as a list of lines
        Returns:
            None        
        '''
        self.doc_ = self._doc_to_list(doc)
        self._verso_analysis()
        self._udpipe_analysis()
        self._parse_responses()
        self._delexicalize()


    def _doc_to_list(self, doc:str|list):
        '''
        Transform an input document to a list of lines (if submitted as string).
        Trim every line.
        ---
        Params:
            doc (str|list) : input document passed either as a string or as
                             a list of lines
        Returns:
            doc (list)     : list of trimmed lines
        '''
        if isinstance(doc, str):
            doc = re.sub('\n[ \t]\n', '\n', doc)
            doc = re.sub('\n+', '\n', doc).strip()
            doc = doc.split('\n')             
        doc = [unicodedata.normalize('NFC', x) for x in doc]
        doc = [x.replace(chr(160), ' ') for x in doc] # Non-breaking space
        doc = [re.sub(' +', ' ', x) for x in doc]
        return [x.strip() for x in doc]


    def _verso_analysis(self):
        '''
        Perform versification analysis with Ingram API
        Response is stored in self.response_ingram_
        ---
        Params:
            None
        Returns:
            None
        '''    
        
        input_data = {'doc': '\n'.join(self.doc_)}
        response = requests.post(self.base_url_ingram_, input_data)
        self.response_ingram_ = json.loads(response.text)


    def _udpipe_analysis(self):
        '''
        Perform tokenization, lemmatization & POS-tagging with UDPipe API
        Response is stored in self.response_ingram_
        ---
        Params:
            None
        Returns:
            None
        '''    
        
        input_data = {
            'tokenizer': '',
            'tagger': '',
            'parser': '',
            'data': ' '.join(self.doc_)
        }
        response = requests.post(self.base_url_udpipe_, input_data)
        self.response_udpipe_ = json.loads(response.text)


    def _parse_responses(self):
        '''
        Parse responses from both Ingram and UDPipe into a single
        list self.doc_parsed_ with a structure 
        [
            {
                'text'      : text of line,
                'bitstring' : bitstring representing stressed/unstressed syllables,
                'meter'     : meter of the line  
                'words'     : list of line's words holding their form, lemma and UPOS
            },
            ...
        ]
        ---
        Params:
            None
        Returns:
            None        
        '''
        word_list = self._conllu_to_list()
        meter = self.pick_meter()       
        self.doc_parsed_ = []
        for text,bitstring in zip(self.doc_, self.response_ingram_['bitstrings']):
            self.doc_parsed_.append({
                'text'      : text,
                'bitstring' : bitstring,
                'meter'     : f'{meter}{len(bitstring)}',
                'words'     : [],
            })
        line_i = 0
        for word in word_list:
            if re.match('^ *$', self.doc_[line_i]):
                line_i += 1   
            if re.match('^ *' + re.escape(word['form']), self.doc_[line_i]):
                self.doc_[line_i] = re.sub('^ *' + re.escape(word['form']), '', self.doc_[line_i])
                self.doc_parsed_[line_i]['words'].append(word)
            else:
                print(word)
                print('UDPIPE:')
                print(f'    »{re.escape(word["form"])}«')
                print('   ', [ord(char) for char in word["form"]])
                print('INPUT:')
                print(f'    »{self.doc_[line_i]}«')
                print('   ', [ord(char) for char in self.doc_[line_i]])
                raise Exception('UDPipe mismatch')


    def _conllu_to_list(self):
        '''
        Transform CoNLL-U (string) into a list of words holding 
        their form, lemma and UPOS
        ---
        Params:
            None
        Returns:
            word_list (list) : List of dicts with keys ('form','lemma','upos')
                               representing individual words      
        '''
        word_list = []
        multiword = None
        for line in self.response_udpipe_['result'].split('\n'):
            fields = line.split('\t')
            if (line.startswith('#') or line == ''):
                continue
            if '-' in fields[0]:
                multiword = fields[0].split('-')[-1]
                mw_form   = fields[1]
                continue
            if multiword is not None:
                if multiword == fields[0]:
                    fields[1] = mw_form
                    multiword = None
                else:
                    fields[1] = ''
            word_list.append({
                'form'    : fields[1],
                'lemma'   : fields[2],
                'upos'    : fields[3],
            })    
        return word_list


    def pick_meter(self):
        '''
        Pick the best scoring meter for input text.
        Returns None if best scoring is below self.min_score_ threshold
        ---
        Params:
            None
        Returns:
            max_meter (str|None) : character representing meter: (i)amb, 
                                   (t)rochee, (d)actyl, (a)mphibrach
        '''
        max_meter = max(
            self.response_ingram_['scores'], 
            key=self.response_ingram_['scores'].get
        )
        if self.response_ingram_['scores'][max_meter] >= self.min_score_:
            return max_meter
        else:
            return None


    def _delexicalize(self):
        '''
        Delexicalize parsed input document.
        Add key 'delex' to each word in self.doc_parsed_ that will hold UPOS in case of
        content words and lemma in case of other
        ---
        Params:
            None
        Returns:
            None
        '''        
        for i,line in enumerate(self.doc_parsed_):
            for j, word in enumerate(line['words']):
                if word['upos'] in ('NOUN', 'ADJ', 'VERB', 'ADV', 'NUM', 'PROPN', 'X'):
                    self.doc_parsed_[i]['words'][j]['delex'] = word['upos']
                else:
                    self.doc_parsed_[i]['words'][j]['delex'] = word['lemma']


    # ***********************************************************************************************************
    # DOCUMENT(S) / DATASET SUMMARY
    # ***********************************************************************************************************


    def doc_summary(self, verbose:bool=True):
        '''
        Either print summary on processed input document or return it as a dict
        ---
        Params:
            verbose (bool) : true = print, false = return dict
        Returns:
            None | dict
        '''
        if verbose:
            print(f'# of lines  : {len(self.doc_parsed_)}')
            print(f'# of tokens : {sum([len(x["words"]) for x in self.doc_parsed_])}')   
            print(f'  meters    : {dict(Counter(d["meter"] for d in self.doc_parsed_))}')     
        else:
            return {
                'n_lines' : len(self.doc_parsed_),
                'n_tokens': sum([len(x["words"]) for x in self.doc_parsed_]),
                'meters'  : {dict(Counter(d["meter"] for d in self.doc_parsed_))},
            }


    def available_candidates(self, meter:str, sample_size:int, verbose:bool=True):
        '''
        Either print number of available candidates and their respective samples for
        selected meter and sample size or return it as a dict
        ---
        Params:
            meter       (str)  : meter + syllable length (e.g. i10 == 10 syllable iamb)
            sample_size (int)  : training data sample size in number of lines, available: 50, 100, 200
            verbose     (bool) : True = print, False = return dict; default: True
        Returns:
            None | dict
        '''
        if str(sample_size) not in self.samplelist_:
            raise Exception(f'Sample size {sample_size} not available. Please use {[int(x) for x in self.samplelist_.keys()]}')
        if meter not in self.samplelist_[str(sample_size)]:
            raise Exception(f'Meter {meter} not available for sample_size == {sample_size}. Available meters: {list(self.samplelist_[sample_size].keys())}')
        if verbose:
            print(f'# of candidates : {len(self.samplelist_[str(sample_size)][meter])}')
            print(f'# of samples    : {self.samplelist_[str(sample_size)][meter]}')
        else:
            return self.samplelist_[str(sample_size)][meter]
        

    # ***********************************************************************************************************
    # CLASSIFICATION
    # ***********************************************************************************************************


    def guess_author( 
        self, 
        sample_size:int, 
        meter:str,
        v_ngram:int|None     = 3,
        w_ngram:int|None     = 1,
        candidate_set:list   = [],
        level_authors:bool   = False,
        random_seed:int|None = None,
        zscores:bool         = True,
        custom_classifier    = None,
    ):
        '''
        Predict author of input document with Support Vector Machine or any custom classifier
        ---
        Params:
            sample_size   (int)       : training data sample size in number of lines
            meter         (str)       : meter + syllable length (e.g. i10 == 10 syllable iamb)
            v_ngram       (int|None)  : how to operationalize versification: 1 == entire bitstrings / 3 == rhythmic 3grams / None; default: 3
            w_ngram       (int|None)  : how to operationalize linguistic level: word (1/2/3)-grams / None; default: 1
            candidate_set (list)      : list of author names to include into candidate set, empty == all available authors; default: []
            level_authors (bool)      : whether to level number of samples per each author; default: False
            random_seed   (int|None)  : optional random seed for leveling authors
            zscores       (bool)      : whether to perform z-score transformation
            custom_classifier         : scikit-learn classifier implementing the 'fit' method; if None, SVC(kernel='linear', C=1) will be created internally.
        Returns:
            verdict       (str)       : predicted author
        '''
        self._extract_features(meter, v_ngram, w_ngram)
        self._load_dataset(sample_size, meter, v_ngram, w_ngram)
        if len(candidate_set) > 1:
            self._reduce_candidate_set(candidate_set)
        if level_authors:
            self._level_authors(random_seed)
        self._concat_datasets()
        if zscores:
            self._zscore_transformation()
        verdict = self._classify_target(custom_classifier)
        return verdict

    
    def _extract_features(self, meter:str, v_ngram:int|None, w_ngram:int|None):
        '''
        Extract selected features from a meter-determined subset of the input document
        Save them into self.target_features_
        ---
        Params:        
            meter         (str)       : meter + syllable length (e.g. i10 == 10 syllable iamb)
            v_ngram       (int|None)  : how to operationalize versification: 1 == entire bitstrings / 3 == rhythmic 3grams / None; default: 3
            w_ngram       (int|None)  : how to operationalize linguistic level: word (1/2/3)-grams / None; default: 1
        Returns:
            None
        '''
        selected_lines = [x for x in self.doc_parsed_ if x['meter'] == meter]
        features = dict()
        if v_ngram == 1:
            features['vf'] = [x['bitstring'] for x in selected_lines]
        elif v_ngram == 3:
            features['v3'] = []
            for line in selected_lines:
                for i in range(len(line['bitstring'])-2):
                    features['v3'].append(line['bitstring'][i:i+3])
        if w_ngram in (1,2,3):
            features[f'w{w_ngram}'] = []
            for line in selected_lines:
                for i in range(len(line['words'])-w_ngram+1):
                    features[f'w{w_ngram}'].append(' '.join([x['delex'] for x in line['words'][i:i+w_ngram]]))
        relative_freqs = dict()
        for type_ in features:
            counts = Counter(features[type_])
            for item,count in counts.items():
                relative_freqs[(type_, item)] = count / len(features[type_])
        self.target_features_ = pd.DataFrame({('_target_', 1): relative_freqs}).T


    def _load_dataset(self, sample_size:int, meter:str, v_ngram:int|None, w_ngram:int|None):
        '''
        Load meter-determined dataset and reduce it to selected feature sets
        Dataset is stored to self.dataset_
        ---
        Params:        
            sample_size   (int)       : training data sample size in number of lines
            meter         (str)       : meter + syllable length (e.g. i10 == 10 syllable iamb)
            v_ngram       (int|None)  : how to operationalize versification: 1 == entire bitstrings / 3 == rhythmic 3grams / None; default: 3
            w_ngram       (int|None)  : how to operationalize linguistic level: word (1/2/3)-grams / None; default: 1
        Returns:
            None
        '''
        self.dataset_ = pd.read_pickle(self.dataset_dir_ / f'{meter}_{sample_size}.pkl')
        cols_to_keep = []
        if v_ngram == 1:
            cols_to_keep.append('vf')
        elif v_ngram == 3:
            cols_to_keep.append('v3')
        if w_ngram is not None:
            cols_to_keep.append(f'w{w_ngram}')
        self.dataset_ = self.dataset_.loc[:, self.dataset_.columns.get_level_values(0).isin(cols_to_keep)]


    def _reduce_candidate_set(self, candidate_set:list):
        '''
        Reduce self.dataset_ to selected authors
        ---
        Params:
            candidate_set (list) : list of author names to include into candidate set, empty == all available authors; default: []
        Returns:
            None
        '''
        self.dataset_ = self.dataset_.loc[self.dataset_.index.get_level_values(0).isin(candidate_set)]


    def _level_authors(self, random_seed:int|None):
        '''
        Level authors so that each of them has the same number of samples
        ---
        Params:
            random_seed   (int|None)  : optional random seed for leveling authors
        Returns:
            None
        '''
        groups = [g for _, g in self.dataset_.groupby(level=0)]
        min_size = min(len(g) for g in groups)
        sampled = [g.sample(n=min_size, random_state=random_seed) for g in groups]
        self.dataset_ = pd.concat(sampled).sort_index()
    

    def _zscore_transformation(self):
        '''
        Perform z-score transformation of self.dataset_
        ---
        Params:
            None
        Returns:
            None
        '''   
        self.dataset_ = (self.dataset_ - self.dataset_.mean()) / self.dataset_.std(ddof=0)     

    
    def _concat_datasets(self):
        '''
        Concat self.dataset_ with features of the input document
        ---
        Params:
            None
        Returns:
            None
        '''  
        self.dataset_ = pd.concat([self.dataset_, self.target_features_]).fillna(0)

    
    def _classify_target(self, custom_classifier):
        '''
        Classify input document
        ---
        Params:
            custom_classifier : scikit-learn classifier implementing the 'fit' method; if None, SVC(kernel='linear', C=1) will be created internally.
        Returns:
            verdict (string)  : predicted author       
        '''
        target_idx = ('_target_', 1)
        X_all = self.dataset_.select_dtypes(include="number").fillna(0)
        y = pd.Series(self.dataset_.index.get_level_values(0), index=self.dataset_.index, name="label")
        X_train = X_all.drop(index=[target_idx])
        y_train = y.drop(index=[target_idx])
        X_target = X_all.loc[[target_idx]]
        if custom_classifier is not None:
            clf = custom_classifier
        else:
            clf = SVC(kernel='linear', C=1)
        clf.fit(X_train, y_train)
        return clf.predict(X_target)[0]