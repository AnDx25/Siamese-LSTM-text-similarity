import pandas as pd
import sys
import tensorflow as tf
import keras
from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist

# File paths

TEST_CSV = './data/test-20.csv'

# Load training set
test_df = pd.read_csv(TEST_CSV)
for q in ['question1', 'question2']:
    test_df[q + '_n'] = test_df[q]

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20


row1=[]
row2=[]
sys.stdout=open("MY_Prediction","w");
test_df, embeddings,row1,row2 = make_w2v_embeddings(test_df, embedding_dim=embedding_dim, empty_w2v=False)


print("embeddings=",embeddings)
# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, max_seq_length)

# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape

# --

model = keras.models.load_model('./data/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})
model.summary()

prediction = model.predict([X_test['left'], X_test['right']])


for i in range(len(prediction)):
	print ("Sentiment1=",row1[i])
	print("sentiment2=",row2[i])
	print("distance=",prediction[i])	












