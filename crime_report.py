import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import streamlit as st

states={'Andaman & Nicobar Islands':0, 'Andhra Pradesh':1, 'Arunachal Pradesh':2,
       'Assam':3, 'Bihar':4, 'Chandigarh':5, 'Chhattisgarh':6,
       'Dadra & Nagar Haveli':7, 'Delhi':8, 'Goa':9, 'Gujarat':10, 'Haryana':11,
       'Himachal Pradesh':12, 'Jammu & Kashmir':13, 'Jharkhand':14, 'Karnataka':15,
       'Kerala':16, 'Madhya Pradesh':17, 'Maharashtra':18, 'Manipur':19, 'Meghalaya':20,
       'Mizoram':21, 'Nagaland':22, 'Odisha':23, 'Puducherry':24, 'Punjab':25,
       'Rajasthan':26, 'Sikkim':27, 'Tamil Nadu':28, 'Tripura':29, 'Uttar Pradesh':30,
       'Uttarakhand':31, 'West Bengal':32}

def bin_encoder(da):
    encoder=ce.BinaryEncoder(cols=["Area_Name"],return_df=True)
    da=encoder.fit_transform(da)
    return da

def clust(df):
    wcss=[]
    for i in range(1,11):
        kmeans=KMeans(n_clusters=i,init="k-means++",random_state=42)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,11),wcss)
    plt.show()

def pred(df):
    k=KMeans(n_clusters=3,init="k-means++",random_state=42)
    y=k.fit_predict(df)
    return y

def dec_to_bin(num):
    li=[]
    for i in (bin(num)[2:].zfill(6)):
        li.append(int(i))
    return li

def prediction(dataset,name,yearx,casesx):
    x=dataset.iloc[:,:-1].values
    y=dataset.iloc[:,-1].values.reshape(-1,1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
    classifier.fit(x_train,y_train)
    arr=[]
    arr+=dec_to_bin((states[(str)(name)]+1))
    year=int(yearx)
    cases=int(casesx)
    arr+=[year,cases]
    return classifier.predict([arr])

data2=pd.read_csv("Indian_crime.csv")
vic2=data2.iloc[:,[0,1,2]]
auto2=data2.iloc[:,[0,1,4]]
mud2=data2.iloc[:,[0,1,3]]
sto2=data2.iloc[:,[0,1,5]]
data2=data2.iloc[:,[0,1,2,3,4,5,6]]

sto=pd.read_csv("10_Property_stolen_and_recovered.csv")
sto=sto[sto["Group_Name"]=="Total Property"]
sto=sto[(sto["Area_Name"]!="Daman & Diu") & (sto["Area_Name"]!="Lakshadweep")]
sto=sto[["Area_Name","Year","Cases_Property_Stolen"]]
sto.rename(columns = {'Cases_Property_Stolen':'Theft_Cases'},inplace=True)

vic=pd.read_csv("20_Victims_of_rape.csv")
vic=vic[vic["Subgroup"]=="Total Rape Victims"]
vic=vic[(vic["Area_Name"]!="Daman & Diu") & (vic["Area_Name"]!="Lakshadweep")]
vic=vic[["Area_Name",'Year','Rape_Cases_Reported']]
vic.rename(columns = {'Rape_Cases_Reported':'Rape_Cases'},inplace=True)

auto=pd.read_csv("30_Auto_theft.csv")
auto=auto.fillna(0)
auto=auto[auto['Group_Name']=="AT6-Total"]
auto=auto[(auto["Area_Name"]!="Daman & Diu") & (auto["Area_Name"]!="Lakshadweep")]
auto=auto[["Area_Name",'Year','Auto_Theft_Stolen']]
auto.rename(columns = {'Auto_Theft_Stolen':'Auto_Theft'},inplace=True)

mud=pd.read_csv("32_Murder_victim_age_sex.csv")
mud=mud.fillna(0)
mud=mud[mud["Sub_Group_Name"]=='3. Total']
mud=mud[(mud["Area_Name"]!="Daman & Diu") & (mud["Area_Name"]!="Lakshadweep")]
mud=mud[["Area_Name",'Year','Victims_Total']]
mud.rename(columns = {'Victims_Total':'Murder_Cases'},inplace=True)

data=vic.merge(mud,on=["Area_Name","Year"])
data=data.merge(auto,on=["Area_Name","Year"])
data=data.merge(sto,on=["Area_Name","Year"])
data["Total_Cases"]=data["Rape_Cases"]+data["Murder_Cases"]+data["Auto_Theft"]+data["Theft_Cases"]

df=data.replace({'Area_Name':states})

all_cases=bin_encoder(df).iloc[:,:].values  
rape=bin_encoder(vic).iloc[:,:].values  
murder=bin_encoder(mud).iloc[:,:].values  
auto_theft=bin_encoder(auto).iloc[:,:].values 
theft=bin_encoder(sto).iloc[:,:].values

mud=bin_encoder(mud)
sto=bin_encoder(sto)
auto=bin_encoder(auto)
vic=bin_encoder(vic)
data=bin_encoder(df)

mud["Group"]=pred(murder)
mud["Group"]=mud["Group"].replace({0:"low",1:"moderate",2:"high"})

mud2["Group"]=pred(murder)
mud2["Group"]=mud2["Group"].replace({0:"low",1:"moderate",2:"high"})

auto["Group"]=pred(auto_theft)
auto["Group"]=auto["Group"].replace({0:"low",1:"moderate",2:"high"})

auto2["Group"]=pred(auto_theft)
auto2["Group"]=auto2["Group"].replace({0:"low",1:"moderate",2:"high"})

data["Group"]=pred(all_cases)
data["Group"]=data["Group"].replace({0:"moderate",1:"low",2:"high"})
data=data.iloc[:,[0,1,2,3,4,5,6,11,12]]

data2["Group"]=pred(all_cases)
data2["Group"]=data2["Group"].replace({0:"moderate",1:"low",2:"high"})

sto["Group"]=pred(theft)
sto["Group"]=sto["Group"].replace({0:"low",1:"moderate",2:"high"})

sto2["Group"]=pred(theft)
sto2["Group"]=sto2["Group"].replace({0:"low",1:"moderate",2:"high"})

vic["Group"]=pred(rape)
vic["Group"]=vic["Group"].replace({0:"low",1:"high",2:"moderate"})

vic2["Group"]=pred(rape)
vic2["Group"]=vic2["Group"].replace({0:"low",1:"high",2:"moderate"})

years=list(range(1990,2050))

st.title("CRIME RATES IN INDIA")
st.subheader("TOTAL CRIMES")
st.dataframe(data2)
col1,col2,col3,col9=st.columns(4)
col1.write("TOTAL RAPE CASES")
col1.dataframe(vic2)
col2.write("TOTAL PROPERTY THEFT CASES")
col2.dataframe(sto2)
col3.write("TOTAL VECHILES THEFT CASES")
col3.dataframe(auto2)
col4,col5=st.columns(2)
col9.write("TOTAL MURDER CASES")
col9.dataframe(mud2)
col4.subheader("FUTURE PREDICTOR")
col4.caption("PREDICT THE CATEGORY OF STATE IN DIFFERENT ASPECTS")
col6,col7,col8=col4.columns(3)
col5.subheader("RESULTS OF CATEGORY OF STATE")
state0=col6.selectbox("STATE NAME",options=data2["Area_Name"].unique(),index=0)
year0=col7.selectbox("YEAR",options=years,index=0)
cases1=col8.number_input("RAPE CASES",step=1)
cases2=col6.number_input("THEFT CASES",step=1)
cases3=col7.number_input("AUTOMOBILE THEFT CASES",step=1)
cases4=col8.number_input("MURDER CASES",step=1)
cases0=cases1+cases2+cases3+cases4
col7.metric("TOTAL CASES",(cases1+cases2+cases3+cases4))
if col7.button("PREDICT"):
    col5.success(state0+" will have "+prediction(dataset=data,name=state0,yearx=year0,casesx=cases0)+" total cases rate in "+str(year0))
    col5.success(state0+" will have "+prediction(dataset=vic,name=state0,yearx=year0,casesx=cases1)+" rape cases rate in "+str(year0))
    col5.success(state0+" will have "+prediction(dataset=sto,name=state0,yearx=year0,casesx=cases2)+" property theft rate in "+str(year0))
    col5.success(state0+" will have "+prediction(dataset=auto,name=state0,yearx=year0,casesx=cases3)+" automobile theft rate in "+str(year0))
    col5.success(state0+" will have "+prediction(dataset=mud,name=state0,yearx=year0,casesx=cases4)+" murder cases rate in "+str(year0))

col10,col11=st.columns(2)
st.subheader("TOTAL CASES OF "+state0.upper())
data2=data2[data2["Area_Name"]==state0].set_index("Year")
data3=data2[["Total_Cases"]]
st.line_chart(data3,use_container_width=True)

col10.subheader("RAPE CASES OF "+state0.upper())
vic2=vic2[vic2["Area_Name"]==state0].set_index("Year")
vic3=vic2[["Rape_Cases"]]
col10.bar_chart(vic3,use_container_width=True)

col11.subheader("THEFT CASES OF "+state0.upper())
sto2=sto2[sto2["Area_Name"]==state0].set_index("Year")
sto3=sto2[["Theft_Cases"]]
col11.bar_chart(sto3,use_container_width=True)

col12,col13=st.columns(2)
col12.subheader("VECHILES THEFTS OF "+state0.upper())
auto2=auto2[auto2["Area_Name"]==state0].set_index("Year")
auto3=auto2[["Auto_Theft"]]
col12.bar_chart(auto3,use_container_width=True)

col13.subheader("MURDER CASES OF "+state0.upper())
mud2=mud2[mud2["Area_Name"]==state0].set_index("Year")
mud3=mud2[["Murder_Cases"]]
col13.bar_chart(mud3,use_container_width=True)