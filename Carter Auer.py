import spacy
from spacy import displacy
from nltk.corpus import reuters

"""
Carter Auer
This code is for the Take home exam and has been ported over from the code provided on canvas
"""




documents = [reuters.raw(doc) for doc in reuters.fileids()][0:100]

# Load the language model
#spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")

# nlp function returns an object with individual token information, 
# linguistic features and relationships
#Visualize using displacy
sentence = documents[1]#'Mona Lisa, the quick brown fox, jumps over Ruffle, the lazy dog'
doc = nlp(sentence)
html = displacy.render(doc, style='dep', options={'distance': 120})
with open("html_parse.html", "w") as file:
   file.write(html)


"""
This example finds organization and countries and the actions they have taken (=they are the SUBJECT)
or actions that were done to them (=they are the OBJECT).
"""

# adjust the pipeline to merge noun phrases and entities into a single token
# https://spacy.io/api/pipeline-functions
nlp.add_pipe("merge_entities")
nlp.add_pipe("merge_noun_chunks")

for doc in nlp.pipe(documents):
    print("="*50)
    for token in doc:
        '''
         (Note there are not docs that fall into Product or Event)
         I created a third or so i ca get more data since rightnow I want to find data about money and its involvement in countries
         and adding ORG will bring in some more relevant information 

'''
        if token.ent_type_ == "GPE" or token.ent_type_ == "MONEY" or token.ent_type_ == "ORG" :#GPE and ORG are entitly labels
            #If the entity is the object
            #print(token)
            #print(token.head)

            if token.dep_ in ("prep"): #orginally dobj
                #tried using prep since i want to gather data on what governments and comapnies are doing 
                #Find the subject (who performed the action)
                subj = [w for w in token.head.lefts if w.dep_ == "nsubj"] #orginally nsubj
                #What action was taken
                verb = token.head
                if subj:
                    print(subj, verb, token)
            #If the entity is the subject

            elif token.dep_ in ("nsubj"): #orginally nsubj
                #Find the object
                obj = [w for w in token.head.rights if w.dep_ == "dobj"] #orginally dobj
                #tried using prep since I want to gather data on what governments and comapnies are doing however I reverted it back based on results
                #Find the action taken
                verb = token.head
                if obj:#changed subj to obj
                    print(token, verb, obj)

