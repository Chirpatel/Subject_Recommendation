#########
# To see the response of similar subject open url https://Subject-Recommendation.chir0313.repl.co   in new tab after clicking the start button.
# It may take some time to install libraries
#########


import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
dc = pd.read_csv("Subjects.csv")
dc= dc[['ID','Subject ID']]
ratings = pd.read_csv("Data.csv",index_col=0)
ratings=ratings.fillna(0)

def standardize(row):
    new_row = (row - row.mean())/(row.max()-row.min())
    return new_row
ratings_std = ratings.apply(standardize)
ratings_std=ratings_std.fillna(0)
item_similarity = cosine_similarity(ratings_std.T)
item_similarity_df = pd.DataFrame(item_similarity,index=ratings.columns,columns=ratings.columns)
item_similarity_df



def get_similar_subjects(Subject_id,user_rating):
    similar_score = item_similarity_df[Subject_id]*(user_rating-ratings.loc[:,Subject_id].mean())
    #similar_score = similar_score.sort_values(ascending=False)
    return similar_score




def item1(id):
    return dc.loc[dc['ID']==id]['Subject'].tolist()[0]




def type1(Subject_id,Ratings):
    recs=get_similar_subjects(Subject_id,Ratings)
    #print(recs)
    l4=[]
    for idx,row in recs.to_frame().iterrows():
        l3=[]
        l3.append(float(row[0]))
        l3.append(int(idx))
        l4.append(l3)
    return l4



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
ds = pd.read_csv("Subjects.csv")
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
ds['Modules']=ds['Modules'].str.replace("-", " ", case = False)
tfidf_matrix = tf.fit_transform(ds['Modules'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
results1 = {}

for idx, row in ds.iterrows():
   similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
   similar_items = [[ds['ID'][i],cosine_similarities[idx][i]] for i in similar_indices] 
   results1[row['ID']] = similar_items[1:]

tfidf_matrix = tf.fit_transform(ds['SLO'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
results2 = {}

for idx, row in ds.iterrows():
   similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
   similar_items = [[ds['ID'][i],cosine_similarities[idx][i]] for i in similar_indices] 
   results2[row['ID']] = similar_items[1:]

tfidf_matrix = tf.fit_transform(ds['Expected Outcome'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
results3 = {}

for idx, row in ds.iterrows():
   similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
   similar_items = [[ds['ID'][i],cosine_similarities[idx][i]] for i in similar_indices] 
   results3[row['ID']] = similar_items[1:]

def item(id):  
    return ds.loc[ds['ID']==id]['Subject Name'].tolist()[0]

def item1(id):
    return dc.loc[dc['ID']==id]['Subject ID'].tolist()[0]

def recommend(item_id, num):  
    recs1=results1[item_id][:]
    recs1.sort()
    recs2=results2[item_id][:]
    recs2.sort()
    recs3=results3[item_id][:]
    recs3.sort()
    recs = []
    for i in range(len(recs1)):
        l=[]
        l.append(recs1[i][0])
        l.append(recs1[i][1])
        l.append(recs2[i][1])
        l.append(recs3[i][1])
        l.append(recs1[i][1]+recs2[i][1]+recs3[i][1])
        recs.append(l)
    recs.sort()
    return recs

def type2(Subject_id):
    return recommend(int(Subject_id), num=1)





def f(Subject_id,Ratings):
    r1=type1(Subject_id,Ratings)
    r2=type2(Subject_id)
    #print("-------")
    #print(r2)
    r2 = r2[0:int(Subject_id)-1]+[[int(Subject_id),0.0,0.0,0.0,0.0]]+r2[int(Subject_id)-1:]
    #print(r2)
    return r1,r2


def item2(s):
		l=dc.loc[dc['Subject ID']==s]['ID'].tolist();
		if(len(l)>0):
			return l[0]
		else:
			return -1;
def f1(subject,ratings,type):
		if(item2(subject)==-1):
			return "-1","-1";
		[r1,r2]=f(str(item2(subject)),int(ratings))
		l4=[]
		for x,y in dc.iterrows():
				l3=[]
				l3.append(int(y[0]))
				l3.append(str(y[1]))
				l3.append(float(r1[int(y[0])-1][0]))
				l3.append(float(r2[int(y[0])-1][1]))
				l3.append(float(r2[int(y[0])-1][2]))
				l3.append(float(r2[int(y[0])-1][3]))
				l3.append(float(r1[int(y[0])-1][0]))
				l3.append(float(r2[int(y[0])-1][4]))
				l3.append(float(r1[int(y[0])-1][0])+float(r2[int(y[0])-1][4]))
				l4.append(l3)
		df = pd.DataFrame(l4, columns = ['ID', 'Subject','Score 1','Score 2','Score 3','Score 4',"Collabrative Score","Content Score","Total Score"])
		if(type == 1):
				df1=df.sort_values(by=["Content Score"],ascending=False)
		elif(type == 3):
				df1=df.sort_values(by=["Total Score"],ascending=False)
		else:
				df1=df.sort_values(by=["Collabrative Score"],ascending=False)
		print(df1.iloc[0])
		Subject = df1.iat[0,1]
		Subjectname = item(df1.iat[0,0] )
		if(Subject == subject):
				Subject = df1.iat[1,1]
				Subjectname = item(df1.iat[1,0] )
		return Subject,Subjectname


def recm(subject,ratings,type):
    [r1,r2]=f1(str(subject).upper(),int(ratings),int(type))
    
    return r1,r2


import string
from flask import Flask, render_template, request, jsonify
import numpy as np
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(  # Create a flask app
	__name__,
	template_folder='templates',  # Name of html file folder
	static_folder='static'  # Name of directory for static files
)

ok_chars = string.ascii_letters + string.digits


@app.route('/')  # What happens when the user visits the site
def base_page():
	return render_template(
		'index.html',  # Template file path, starting from the templates folder. 
	)

@app.route("/",methods=["POST"])
def recommend1():
		message = request.get_json(force=True)
		print(message)
		subjectcode,subjectname=recm(message['subjectid'],message['ratings'],message['type'])
		response = {
				'subjectname': str(subjectname),
				'subjectcode': str(subjectcode)

		}
		print("Recommendate:",response['subjectname'])
		print("Response:",response)
		return jsonify(response)



if __name__ == "__main__":  # Makes sure this is the main process
	app.run( # Starts the site
		host='0.0.0.0',  # EStablishes the host, required for repl to detect the site
		port=5000,  # Randomly select the port the machine hosts on.
		threaded=False
	)







