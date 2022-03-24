import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import docx2txt
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

####################################################################
#####################  FUNCTION DEFINITION  ########################
####################################################################

# create a function to convert the text to lower case
def text_to_lowerCase(text):
    text_lowerCase = text.lower() # convert the text to lower case
    return text_lowerCase

# create a function to remove all the punctuations from the text
def removes_punctuation(text):
    text_no_punct = re.sub(r'[^\w\s]', '', text) # using regexp to remove all the special characters and punctuations from the text 
    return text_no_punct

# create a function for tokenization process
def word_tokenization(text):
    text_tokenization = word_tokenize(text) # split the text into a list of tokens 
    return text_tokenization

# create a function to remove stopwords from the text
def remove_stopwords(text):
    
    # initialize an empty word list 
    newWordList = list()
    
    for word in text:
        if word not in stopwords.words('english'): # remove stopwords from the list of tokens 
            newWordList.append(word)    
    return newWordList


# create a function to lemmatize the words from the text
def word_lemmatize(text):
    
    # define an object of lemmatization 
    lemmatizer = WordNetLemmatizer()
    
    lemmatizeWordList = list()
    for word in text:
        lemmatizeWordList.append(lemmatizer.lemmatize(word)) # lemmatize list of words 
    return lemmatizeWordList
        

# create a function to form an embedding vector for respective document 
def embeddingVector(text, keywords):
    # initialize an empty vector list
    vector = list()
    
    # create an embedding vector space 
    for word in keywords:
        if word in text:
            vector.append(1)
        else:
            vector.append(0)
    return vector
    
def computeCosineSimilarity(vectorA, vectorB, keywords):
    # initialization 
    c = 0
    a = 0
    b = 0
    
    for i in range(len(keywords)):
        c+= vectorA[i] * vectorB[i]
        a+= vectorA[i]**2
        b+= vectorB[i]**2
    
    # compute the cosine similarity 
    cosineSimilarity = c/(np.sqrt(a) * np.sqrt(b))
    return cosineSimilarity



#####################################################
#####################  MAIN  ########################
#####################################################

def main():
    
    st.title("Document Similarity")
    st.write("""
                 This app uses the **Cosine Similarity** method to find the similarity between the two text documents.
                 """)
    # prompt the user to upload document A
    st.subheader("Upload Document Files:")
    docA_file = st.file_uploader("Upload Document A:", type = ['docx', 'txt'], accept_multiple_files = False)
    
    # prompt the user to upload document B
    docB_file = st.file_uploader("Upload Document B:", type = ['docx', 'txt'], accept_multiple_files = False)
    
    # prompt the user to click on the button to process the computation 
    clickProcess = st.button("Process")
    
    if clickProcess:
        if docA_file is not None and docB_file is not None:
            
            # step 1: Read the uploaded documents 
            st.header("Step 1: Read the uploaded documents. ")
            
            # read document A
            if docA_file.type == "text/plain":
                docA_raw_text = str(docA_file.read(), "utf-8")
                st.subheader("Raw Text of Document A:")
                st.write(docA_raw_text)
            else:
                docA_raw_text = docx2txt.process(docA_file)
                st.subheader("Raw Text of Document A:")
                st.write(docA_raw_text)
            
            
            # read document B
            if docB_file.type == "text/plain":
                docB_raw_text = str(docB_file.read(), "utf-8")
                st.subheader("Raw Text of Document B:")
                st.write(docB_raw_text)
            else:
                docB_raw_text = docx2txt.process(docB_file)
                st.subheader("Raw Text of Document B: ")
                st.write(docB_raw_text)
            
            
            
            # step 2: Text Cleaning
            st.header("Step 2: Text Cleaning")
            
            # step 2.1: convert the text to lower cases.
            st.subheader("Step 2.1: Transforming the text into lower cases. ")
            
            # Call the function to convert text document A into its lower case
            docA_lower_case_text = text_to_lowerCase(docA_raw_text)
            st.subheader("Document A: ")
            st.write(docA_lower_case_text)
            
            # Call the function to convert text document B into its lower case
            docB_lower_case_text = text_to_lowerCase(docB_raw_text)
            st.subheader("Document B:")
            st.write(docB_lower_case_text)
            
            
            # step 2.2: remove the punctuation from the text
            st.subheader("Step 2.2: Remove all the special characters and punctuations from text. ")
            
            # call the function to remove all the special characters and punctuations from document A
            docA_no_punct_text = removes_punctuation(docA_lower_case_text)
            st.subheader("Document A: ")
            st.write(docA_no_punct_text)    
            
            # call the function to remove all the special characters and punctuations from document B
            docB_no_punct_text = removes_punctuation(docB_lower_case_text)
            st.subheader("Document B: ")
            st.write(docB_no_punct_text)  
            
            
            # step 2.3: tokenization 
            st.subheader("Step 2.3: Splitting the text into a list of tokens (also known as Tokenization). ")
            
            # call the function to split the text in document A into a list of tokens
            docA_token = word_tokenization(docA_no_punct_text)
            st.subheader("List of tokens for Document A: ")
            st.dataframe(sorted(docA_token))
            
            # call the function to split the text in document B into a list of tokens
            docB_token = word_tokenization(docB_no_punct_text)
            st.subheader("List of tokens for Document B: ")
            st.dataframe(sorted(docB_token))
            
            
            # step 2.4: removing stopwords
            st.subheader("Step 2.4: Removing stopwords that are most commonly used in the language and do not add too much meaning to the text.")
            
            # call the function to remove stopwords from text in document A
            docA_no_stopwords = remove_stopwords(docA_token)
            st.subheader("List of tokens with stopwords removed for Document A: ")
            st.dataframe(sorted(docA_no_stopwords))
            
            # call the function to remove stopwords from text in document B
            docB_no_stopwords = remove_stopwords(docB_token)
            st.subheader("List of tokens with stopwords removed for Document B: ")
            st.dataframe(sorted(docB_no_stopwords) )
            
            
            # step 2.5: lemmatization 
            st.subheader("Step 2.5: Getting the same words for a group of inflected word forms (also known as Lemmatization). ")
            
            # call the function to lemmatize the text in document A
            docA_text_lemmatize = word_lemmatize(docA_no_stopwords)
            st.subheader("List of lemmatize words for Document A: ")
            st.text(docA_text_lemmatize)
            
            # call the function to lemmatize the text in document B
            docB_text_lemmatize = word_lemmatize(docB_no_stopwords)
            st.subheader("List of lemmatize words for Document B: ")
            st.text(docB_text_lemmatize)  
            
            # copy the cleaned text into new variables
            docA_cleaned_text = docA_text_lemmatize
            docB_cleaned_text = docB_text_lemmatize
            
            
            # step 3: Create a list of keywords between document A and document B
            st.header("Step 3: Form a set of keywords between Document A and Document B")
            keywords = set(docA_cleaned_text).union(set(docB_text_lemmatize))
            st.dataframe(sorted(keywords))
            
            # step 4: create embedding vector for document A and document B
            st.header("Step 4: Create an embedding vector for Document A and Document B")
            
            # call the function to create embedding vector for document A
            st.subheader("Embedding Vector for Document A:")
            docA_vector = embeddingVector(docA_cleaned_text, keywords)
            st.text(docA_vector)
            
            # call the function to create embedding vector for document B
            st.subheader("Embedding Vector for Document B:")
            docB_vector = embeddingVector(docB_cleaned_text, keywords)
            st.text(docB_vector)
            
            
            # step 5: find the similarity between two documents
            st.header("Step 5: Find the similarity between Document A and Document B: ")
            # call the function to compute the cosine similarity between document A and document B
            cosine_similarity = computeCosineSimilarity(docA_vector, docB_vector, keywords)
            
            st.info("Similarity between Document A and Document B: {:.4f} ({:.4f}%)".format(cosine_similarity, cosine_similarity*100))
            
            # step 6: conclusion 
            st.header("Step 6: Conclusion")
            if cosine_similarity >= 0.8 and cosine_similarity <= 1.00:
                st.success("Both Document A and Document B are highly the same. ")
            elif cosine_similarity >= 0.5 and cosine_similarity < 0.79:
                st.success("Both Document A and Document B are moderately the same. ")
            elif cosine_similarity >= 0.3 and cosine_similarity < 0.49:
                st.success("Both Document A and Document B are moderately not the same. ")
            else:
                st.success("Both Document A and Document are highly not the same. ")
            
            
        elif docA_file is None and docB_file is None:
            st.error("Please upload Document A and Document B in this form of docx. or txt.")
        elif docA_file is None:
            st.error("Please upload Document A in the form of docx. or txt. ")
        else:
            st.error("Please upload Document B in the form of docx. or txt.")

if __name__ == "__main__":
    main()
