# PoetGuesser
Python library tailored for authorship recognition of Czech poetry.
Notebook test.ipynb contains a tutorial showcase targeting part of the text of "Cid v zrcadle španělškých romancí" by Jaroslav Vrchlický (testdoc.txt)

## How it works
Recognition is based on a feature set combining delexicalized linguistic features (frequencies of delexicalized tokens, token bigrams, or token trigrams) and versification features (frequencies of rhythmical bitstrings or rhythmical trigrams). Both linguistic and versification analysis are provided on-the-fly by [UDPipe API](https://lindat.mff.cuni.cz/services/udpipe/api-reference.php) and [Ingram API](https://versologie.cz/v2/tool_ingram/), respectively.

## Key features
- Built-in SVC model, but any kind of sklearn model may be used 
- Pretrained models for a number of Czech poets and most frequent poetic meters (given that poetic meters are of immense effect on vocabulary, recognition is always based on texts written in a single meter)

## Roles
- Petr Plecháč - principal programmer (nebo něco podobného)
 
## Dedication
National Library of the Czech Republic
Realized with the support of institutional research of National Library of the Czech Republic funded by Ministry of Culture of the Czech Republic as part of the framework of Longterm conception developement of scientific organization, DKRVO, 9: Digital Humanities.
Trained on data from the Institute for Czech Literature of the Czech Academy of Sciences.

