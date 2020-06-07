#########
# To see the response of similar subject open url https://Subject-Recommendation.chir0313.repl.co   in new tab after clicking the start button.
# It may take some time to install libraries
#########


import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def rate(Subject_id,Ratings):
	ratings = pd.read_csv("Data.csv",index_col=0)
	ratings=ratings.fillna(0)

	def standardize(row):
			new_row = (row - row.mean())/(row.max()-row.min())
			return new_row
	ratings_std = ratings.apply(standardize)
	ratings_std=ratings_std.fillna(0)
	item_similarity = cosine_similarity(ratings_std.T)
	item_similarity_df = pd.DataFrame(item_similarity,index=ratings.columns,columns=ratings.columns)
	#print(item_similarity_df)
	item_similarity_df

	def get_similar_subjects(Subject_id,user_rating):
			item_similarity = item_similarity_df[Subject_id]
			similar_score = item_similarity_df[Subject_id]*(abs(user_rating-ratings.loc[:,Subject_id].mean()))
			return similar_score.to_numpy(),item_similarity.to_numpy()
	return get_similar_subjects(Subject_id,Ratings)




def item1(id):
		dc = pd.read_csv("Subjects.csv")
		dc= dc[['ID','Subject ID']]
		return dc.loc[dc['ID']==id]['Subject'].tolist()[0]

import csv
from csv import writer
def append_list_as_row( list_of_elem):
		with open('Data.csv', 'a+', newline='') as write_obj:
				csv_writer = writer(write_obj)
				csv_writer.writerow(list_of_elem)
				
def add(ID,index,value):
		df = pd.read_csv("Data.csv")
		df.at[int(ID)-1, str(index)] = value
		df.to_csv("Data.csv", index=False)

def id():
		df = pd.read_csv("Data.csv")
		num = len(df)+1
		l = [''] * len(df.columns)
		l[0] = str(num)
		append_list_as_row(l)
		return l[0]

def func(ID,index,value):
		if(index!=-1):
				add(ID,index,value)



def type1(Subject_id,Ratings):
		rec1,rec2=rate(Subject_id,Ratings)
		l4=[]
		for i in range(len(rec1)):
				l3=[]
				l3.append(int(i+1))
				l3.append(float(rec1[i]))
				l3.append(float(rec2[i]))
				l4.append(l3)
		return l4



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 


def subject():
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

	return results1,results2,results3;

def item(id):  
		ds = pd.read_csv("Subjects.csv")
		return ds.loc[ds['ID']==id]['Subject Name'].tolist()[0]

def item1(id):
		dc = pd.read_csv("Subjects.csv")
		dc= dc[['ID','Subject ID']]
		return dc.loc[dc['ID']==id]['Subject ID'].tolist()[0]

def recommend(item_id, num):  
		results1,results2,results3 = subject()
		recs1=results1[item_id][:]
		recs1.sort()
		recs2=results2[item_id][:]
		recs2.sort()
		recs3=results3[item_id][:]
		recs3.sort()
		#print(recs1,recs2,recs3)
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
		r2 = r2[0:int(Subject_id)-1]+[[int(Subject_id),0.0,0.0,0.0,0.0]]+r2[int(Subject_id)-1:]

		return r1,r2


def item2(s):
		#print(s)
		dc = pd.read_csv("Subjects.csv")
		dc= dc[['ID','Subject ID']]
		l=dc.loc[dc['Subject ID']==s]['ID'].tolist();
		if(len(l)>0):
			return l[0]
		else:
			return -1;
def f1(subject,ratings,type):
		dc = pd.read_csv("Subjects.csv")
		dc= dc[['ID','Subject ID']]
		if(item2(subject)==-1):
			return "-1","-1","-1";
		[r1,r2]=f(str(item2(subject)),int(ratings))
		l4=[]
		for x,y in dc.iterrows():
				l3=[]
				l3.append(int(y[0]))
				l3.append(str(y[1]))
				l3.append(float(r1[int(y[0])-1][2]))
				l3.append(float(r2[int(y[0])-1][1]))
				l3.append(float(r2[int(y[0])-1][2]))
				l3.append(float(r2[int(y[0])-1][3]))
				l3.append(float(r1[int(y[0])-1][1]))
				l3.append(float(r2[int(y[0])-1][4]))
				l3.append(float(r1[int(y[0])-1][1])+float(r2[int(y[0])-1][4]))
				l4.append(l3)
		df = pd.DataFrame(l4, columns = ['ID', 'Subject','Score 1','Score 2','Score 3','Score 4',"Collabrative Score","Content Score","Total Score"])

		if(type == 1):
				df1=df.sort_values(by=["Content Score"],ascending=False)
				percentage = (df1.iat[0,7])/3 *100
		elif(type == 3):
				df1=df.sort_values(by=["Total Score"],ascending=False)
				percentage = (df1.iat[0,7]+df1.iat[0,2])/4 * 100
		else:
				df1=df.sort_values(by=["Collabrative Score"],ascending=False)
				percentage = (df1.iat[0,2]) * 100

		Subject = df1.iat[0,1]
		Subjectname = item(df1.iat[0,0] )
		if(Subject == subject):
				print(df1.iloc[1])
				Subject = df1.iat[1,1]
				Subjectname = item(df1.iat[1,0] )
				if(type==1):
					percentage = (df1.iat[1,7])/3 *100
				elif(type==3):
					percentage = (df1.iat[1,7]+df1.iat[1,2])/4 * 100
				else:
					percentage =(df1.iat[1,2]) * 100
		else:
			print(df1.iloc[0])
		return Subject,Subjectname,percentage


def recm(subject,ratings,type):
		[r1,r2,p]=f1(str(subject).upper(),int(ratings),int(type))
		return r1,r2,p


def add_details(Subject_ID,Subject_Name,Modules,SLO,Expected_Outcome):
		df = pd.read_csv("Data.csv")
		columns = list(df.head(0))
		ID = int(columns[-1])+1
		df[ID] = ""
		df.to_csv("Data.csv", index=False)
		fields=[ID,Subject_ID,Subject_Name,Modules,SLO,Expected_Outcome]
		with open('Subjects.csv', 'a') as f:
				writer = csv.writer(f)
				writer.writerow(fields)
def add_details_csv(Subject_ID,Subject_Name,Modules,SLO,Expected_Outcome):
		fields=[Subject_ID,Subject_Name,Modules,SLO,Expected_Outcome]
		with open('add.csv', 'a') as f:
				writer = csv.writer(f)
				writer.writerow(fields)

import string
from flask import Flask, render_template, request, jsonify
import numpy as np
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(  # Create a flask app
	__name__,
	template_folder='templates',  # Name of html file folder
	static_folder='syllabus'  # Name of directory for static files
)

ok_chars = string.ascii_letters + string.digits


@app.route('/')  # What happens when the user visits the site
def base_page():
	return render_template(
		'index.html',  # Template file path, starting from the templates folder. 
	)
@app.route('/id')  # What happens when the user visits the site
def get_id():
	new_id  = id()
	return jsonify({'id':str(new_id)})
@app.route('/add')  # What happens when the user visits the site
def add_page():
	return render_template(
		'add.html',  # Template file path, starting from the templates folder. 
	)

@app.route('/<path:path>')
def static_file(path):
		return app.send_static_file(path)


@app.route("/",methods=["POST"])
def recommend1():
		message = request.get_json(force=True)
		print(message)
		#print(item2(message['subjectid']))
		func(message['id'],item2(message['subjectid'].upper()),message['ratings'])
		#func(item2(message['subjectid']),message['ratings'])
		subjectcode,subjectname,percentage=recm(message['subjectid'],float(message['ratings']),message['type'])

		response = {
				'subjectname': str(subjectname),
				'subjectcode': str(subjectcode),
				'percentage':percentage
		}
		print("Recommendate:",response['subjectname'])
		print("Response:",response)
		return jsonify(response)

@app.route("/add",methods=["POST"])
def add_subject():
		message = request.get_json(force=True)
		print(message)
		if(message['authkey']=="1234"):
			add_details(message['subjectcode'].upper(),message["subjectname"],message["subjectmodule"],message["slo"],message["subjectoutcome"])
			response = {
					'status': "Added"
			}
		else:
			add_details_csv(message['subjectcode'],message["subjectname"],message["subjectmodule"],message["slo"],message["subjectoutcome"])
			response = {
					'status': "Admin will add this subject!"
			}
		print("Response:",response)
		return jsonify(response)		
			



if __name__ == "__main__":  # Makes sure this is the main process
	app.run( # Starts the site
		host='0.0.0.0',  # EStablishes the host, required for repl to detect the site
		port=5000,  # Randomly select the port the machine hosts on.
		threaded=False
	)








