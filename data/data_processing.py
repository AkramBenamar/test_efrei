import re
import numpy as np
import pandas as pd
import itertools
import os
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

class DataPreprocessor:
    def __init__(self, embedding_path, sample_size=10000):
        self.embedding_path = embedding_path
        self.sample_size = sample_size
        self.stop_words = set(stopwords.words('english'))
        self.word_to_index = None

    def text_to_word_list(self, text):
        # Pre-process and convert texts to a list of words
        text = str(text)
        text = text.lower()

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        text = text.split()
        return text

    def make_w2v_embeddings(self, df, embedding_dim=300, empty_w2v=False):
        vocabs = {}
        vocabs_cnt = 0

        vocabs_not_w2v = {}
        vocabs_not_w2v_cnt = 0

        # Stopwords
        stops = set(stopwords.words('english'))

        # Load word2vec
        print("Loading word2vec model (this may take 2-3 mins) ...")
        word2vec = KeyedVectors.load_word2vec_format(self.embedding_path, binary=True)

        for index, row in df.iterrows():
            # Print the number of embedded sentences
            if index != 0 and index % 1000 == 0:
                print("{:,} sentences embedded.".format(index), flush=True)

            # Iterate through the text of both questions of the row
            for question in ['question1', 'question2']:
                q2n = []  # q2n -> question numbers representation
                for word in self.text_to_word_list(row[question]):
                    # Check for unwanted words
                    if word in stops:
                        continue

                    # If a word is missing from word2vec model
                    if word not in word2vec.key_to_index:
                        if word not in vocabs_not_w2v:
                            vocabs_not_w2v_cnt += 1
                            vocabs_not_w2v[word] = 1

                    # If you have never seen a word, append it to vocab dictionary
                    if word not in vocabs:
                        vocabs_cnt += 1
                        vocabs[word] = vocabs_cnt
                        q2n.append(vocabs_cnt)
                    else:
                        q2n.append(vocabs[word])

                # Append question as number representation
                df.at[index, question + '_n'] = q2n

        embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)  
        embeddings[0] = 0  

        # Build the embedding matrix
        for word, index in vocabs.items():
            if word in word2vec.key_to_index:
                embeddings[index] = word2vec.get_vector(word)
        del word2vec

        return df, embeddings

    def split_and_zero_padding(self, df, max_seq_length):
        # Split to dicts
        X = {'left': df['question1_n'], 'right': df['question2_n']}

        # Zero padding
        for dataset, side in itertools.product([X], ['left', 'right']):
            dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

        return X

    def load_and_process_data(self, train_csv, max_seq_length=20):
        # Load training set
        df = pd.read_csv(train_csv)
        df=df.head(1000)
        for q in ['question1', 'question2']:
            df[q + '_n'] = df[q]

        # Make word2vec embeddings
        embedding_dim = 300
        use_w2v = True

        df, embeddings = self.make_w2v_embeddings(df, embedding_dim=embedding_dim, empty_w2v=not use_w2v)

        # Split to train-validation-test
        train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.1 / 0.8, random_state=42)

        # Split and pad sequences
        X_train = self.split_and_zero_padding(train_df[['question1_n', 'question2_n']], max_seq_length)
        Y_train = train_df['is_duplicate'].values

        X_val = self.split_and_zero_padding(val_df[['question1_n', 'question2_n']], max_seq_length)
        Y_val = val_df['is_duplicate'].values

        X_test = self.split_and_zero_padding(test_df[['question1_n', 'question2_n']], max_seq_length)
        Y_test = test_df['is_duplicate'].values

        # Verify alignment
        assert X_train['left'].shape[0] == len(Y_train)
        assert X_train['right'].shape[0] == len(Y_train)
        assert X_val['left'].shape[0] == len(Y_val)
        assert X_val['right'].shape[0] == len(Y_val)
        assert X_test['left'].shape[0] == len(Y_test)
        assert X_test['right'].shape[0] == len(Y_test)

        return X_train, X_val, Y_train, Y_val, X_test, Y_test, embeddings
