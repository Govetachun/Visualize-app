
import streamlit as st
import pandas as pd
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_juxtapose import juxtapose

from PIL import Image
import requests

import pathlib
if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)
st.set_page_config(layout="centered", page_icon="🔭", page_title="Visualizing app")
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Visualize Dashboard for Deep Learning")
    choice = st.radio("Navigaion",["Input data","Data exploratory","Machine Learning Model","Choose epoch to run ML model","Bonus: Comparing Image"])
    st.info("This is project for deep learning visualization")
col10, col20, col30 = st.columns(3)
col10.metric("Temperature", "70 °F", "1.2 °F")
col20.metric("Wind", "9 mph", "-8%")
col30.metric("Humidity", "86%", "4%")
if choice == "Input data":

        st.info("Random your data as 20x500 and save to .csv file")
        click = st.button("Random your data")
        if click:
            df = pd.DataFrame(np.random.randn(20, 500), columns= list(range(1,501)))

            df.to_csv('dataset.csv',index=None)
            st.dataframe(df)
            st.info(df.shape)

              
elif choice == "Data exploratory":
    dropdown_box = st.selectbox(
        'Choose the chart plot: ',
        ('Bar chart','Line chart', 'Scatter chart',"Print all")
    )
    if dropdown_box != "Print all":
        text_input = st.text_input(
            "Enter the number of chart N 👇",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
        )
        if text_input:
            st.write("The dashboard show: ", text_input,dropdown_box)
            plot_click = st.button("Plot your data")
            if plot_click:
                df3 = df
                df3['index_col'] =df3.index
                if dropdown_box == 'Bar chart':
                    for i in range(int(text_input)):
                        ramd = str(random.randint(0, 500))
                        st.write("Bar chart",i,": Column",ramd)
                        fig = plt.figure(figsize=(4, 4))
                        sns.barplot(data = df3,x = 'index_col',y = ramd)      
                        st.pyplot(fig)
                elif dropdown_box =='Line chart':
                    for i in range(int(text_input)):
                        ramd = str(random.randint(0, 500))
                        st.write("Line chart",i,": Column",ramd)
                        st.line_chart(df[ramd])
                        
                elif dropdown_box =='Scatter chart':
                    fig = [ j for j in range(int(text_input))]
                    ax = [k for k in range(int(text_input))]
                    for i in range(int(text_input)):
                        ramd1 = str(random.randint(0, 500))
                        ramd2 = str(random.randint(0, 500))
                        st.write("Scatter chart",i,": Column",ramd1,"and Colum",ramd2)
                        fig[i],ax[i] = plt.subplots()
                        ax[i].scatter(df[ramd1],df[ramd2])
                        st.pyplot(fig[i])
                st.write("Random N input from dataset")
                st.dataframe(df.sample(n = int(text_input)))
    else:
        # st.set_page_config(layout="wide")
        
        st.title("Gererating 500 data per epochs")
        generate_graph = st.button("Click to start generate data")
        
        if generate_graph:
            df2 = df
            df2['index_col'] =df2.index
            # fig = plt.figure(figsize=(4, 4))
            k = 1
            j = 1 
            l = 1
            for i in range (1,6 ):
                st.write("\nTotal images generated per chart:",i*100)
                
                my_bar = st.progress(0)
                my_bar2 = st.progress(0)
                my_bar3 = st.progress(0)
                col1, col2,col3= st.columns(3)
                with col1:
                                       
                    while 1:         
                            fig = plt.figure(figsize=(4, 4))
                            sns.barplot(data = df2,x = 'index_col',y = str(k))
                            # plt.bar(df[str(k)])
                            fn = './data/bar/bar' +str(k) +'.png'
                            plt.savefig(fn)  
                            with st.expander("Click to see the sample of bar chart"):
                                        st.write("""
                                        This is one sample of bar chart
                                        """)
                                        st.pyplot(fig)

                            my_bar.progress(k%100)
                            k+=1
                            if k == i*100:
                                break
                with col2:
                    while 1:
                            # st.line_chart(df[str(j)])
                            
                            fig = plt.figure(figsize=(4, 4))
                            sns.lineplot(data = df2,x='index_col',y = str(j))
                            fn = './data/line/line' +str(j) +'.png'
                            plt.savefig(fn)
                            # st.pyplot(fig)
                            with st.expander("Click to see the sample of line chart"):
                                        st.write("""
                                            The chart above shows some numbers I picked for you.
                                            I rolled actual dice for these, so they're *guaranteed* to
                                            be random.
                                        """)
                                        st.pyplot(fig)
                            my_bar2.progress(j%100)
                            j +=1
                            if j == i*100:
                                break
                with col3:
                    while 1:
                            rand1 = str(l)
                            rand2 = str(l+1)
                            # my_bar3.progress(l + 1)
                            # fig2,ax2 = plt.subplots()
                            # ax2.scatter(df[rand1],df[rand2])
                            fig = plt.figure(figsize=(4, 4))
                            sns.scatterplot(x=df[rand1], y=df[rand2])
                            fn = './data/scatter/scatter' +str(l) +'.png'
                            plt.savefig(fn)
                            # fig2.savefig("scatter" + i +".png")
                            # st.pyplot(fig)
                            with st.expander("Click to see the sample of scatter Plot"):
                                        st.write("""
                                            The chart above shows some numbers I picked for you.
                                            I rolled actual dice for these, so they're *guaranteed* to
                                            be random.
                                        """)
                                        st.pyplot(fig)
                            my_bar3.progress(l%100)
                            l+=1
                            if l == i*100:
                                break
            st.success("Done")
            
elif choice == "Machine Learning Model":

    st.info("Run the Classification Model")
    compile_button = st.button("Start to compile model")
    if compile_button:
        with st.spinner('Wait for it...'):
            from image_classification import historyfile
            with st.expander("View the training process information"):
                df5 = historyfile()
                st.dataframe(df5)
        
                fig1 = plt.figure(figsize=(4, 4))
                sns.lineplot(data = df5,x = 'epochs',y = 'accuracy')   
                sns.lineplot(data = df5,x = 'epochs',y = 'val_accuracy')   
                st.pyplot(fig1)
                fig4 = plt.figure(figsize=(4, 4))
                sns.lineplot(data = df5,x = 'epochs',y = 'loss')   
                sns.lineplot(data = df5,x = 'epochs',y = 'val_loss')   
                st.pyplot(fig4)
                fig2 = plt.figure(figsize=(4, 4))
                sns.lineplot(data = df5,x = 'epochs',y = 'elaped time')   
                st.pyplot(fig2)
                fig3 = plt.figure(figsize=(4, 4))
                sns.barplot(data = df5,x = 'epochs',y = 'elaped time')   
                st.pyplot(fig3)
        st.success("Compile done")

elif choice == "Choose epoch to run ML model":
    st.info("Choose epoch to run ML model")
    text_input = st.text_input(
            "Enter the number of epoch you want to train 👇",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
        )
    compile_button = st.button("Start to compile model")
    if compile_button:
        with st.spinner('Wait for it...'):
            from image_classification import training_withepoch
            
            with st.expander("View the training process information"):
                from image_classification import print_per_cluster
                training_withepoch(int(text_input))
                st.write("View the first 5 picture of each chart of the",text_input,"epoch")
                tab1, tab2,tab3 = st.tabs(["bar chart","line chart","scatter chart"])
                with tab1:
                    
                    print_per_cluster("./output/train/bar/*.png")
                with tab2:
                    print_per_cluster("./output/train/line/*.png")
                with tab3:
                    print_per_cluster("./output/train/scatter/*.png")
                    
        st.success("Compile done")
        
elif choice == "Bonus: Comparing Image":
    st.title("Comparing 2 image from satellite")
    STREAMLIT_STATIC_PATH = (
        pathlib.Path(st.__path__[0]) / "static"
    )  # at venv/lib/python3.9/site-packages/streamlit/static

    IMG1 = "img1.png"
    IMG2 = "img2.png"

    
    DEFAULT_IMG1_URL = (
        "https://juxtapose.knightlab.com/static/img/Sochi_11April2005.jpg"
    )
    DEFAULT_IMG2_URL = (
        "https://juxtapose.knightlab.com/static/img/Sochi_22Nov2013.jpg"
    )

    def fetch_img_from_url(url: str) -> Image:
        img = Image.open(requests.get(url, stream=True).raw)
        return img

    form = st.form(key="Image comparison")
    img1_url = form.text_input("Image one url", value=DEFAULT_IMG1_URL)
    img1 = fetch_img_from_url(img1_url)
    img1.save(STREAMLIT_STATIC_PATH / IMG1)

    img2_url = form.text_input("Image two url", value=DEFAULT_IMG2_URL)
    img2 = fetch_img_from_url(img2_url)
    img2.save(STREAMLIT_STATIC_PATH / IMG2)

    submit = form.form_submit_button("Submit")
    if submit:
        juxtapose(IMG1, IMG2)
    
