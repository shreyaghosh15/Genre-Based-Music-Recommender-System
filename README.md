# Genre-Based Music Recommendation System
I have used a Machine Learning item-based collaborative filtering approach to recommend songs to a user, based on a crowdsourced data of listen counts of 1 million songs. <br/>
<br/>
Datasets: The songs metadata is from The Million Song Dataset by Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. It consists of metadata of 1 million songs, collected over a period of several years. Another triplets file is used that contains the user-to-song data for those 1 million songs. I have merged the two datasets to create a cross tab of the data required by the algorithm. <br/>
Link to the datasets: http://millionsongdataset.com/pages/getting-dataset/
<br/>
The algorithm I used is Singular Value Decomposition, which is a Matrix Factorization technique. The SVD is followed by finding the Pearson Correlation Coefficient of the resultant matrix to recommend songs to a user based on the input song's genre.
<br/>
I have used Python to implement the project code and built an integrated web application using its Streamlit library.
