import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
import wordmap
import replace_with_synonym
import cv2

def entropy(s):
    p, ln = pd.Series(s).value_counts(normalize=True), np.log2
    return -p.dot(ln(p))


def quasi_identifiers_fn(df, threshold=0.8):
    QI = []
    for col in df.columns:
        E = entropy(df[col])
        if E > threshold:
            QI.append(col)
    return QI


def calculate_entropy(column):
    """
    Calculate the entropy of a column of data.
    """
    value_counts = column.value_counts()
    proportions = value_counts / len(column)
    entropy = -(proportions * np.log2(proportions)).sum()
    return entropy


def calculate_proportion_unique(data, quasi_identifier):
    """
    Calculate the proportion of records with a unique quasi-identifier.
    """
    grouped = data.groupby(quasi_identifier)
    unique_counts = grouped.size().value_counts()
    total = unique_counts.sum()
    proportion_unique = unique_counts[1] / total
    return proportion_unique


def generalize(value, column_name):
    if column_name == 'age':
        return round(value/5.0)*5
    elif column_name == 'zipcode':
        return str(value)[:3]
    else:
        return value

def k_anonymity_algo(df):
    print(list(df.columns))

    print(df.dtypes)
    # Display column headers
    st.write("## Select Quasi Identifiers")

    # Get column names
    columns = df.columns.tolist()

    # Create checkboxes for column selection
    generalize_cols = st.multiselect("Select columns", columns)
    st.write(df)
    quasi_identifiers = quasi_identifiers_fn(df)
    print(len(quasi_identifiers), len(df.columns))
    k_level = st.slider("Select k-anonymity level", 2, 50, 2)
    st.write("Selected k-anonymity level:", k_level)

    quasi_identifiers = generalize_cols

    k = k_level

    sensitive_attr_list = [
        col for col in df.columns if col not in quasi_identifiers]
    sensitive_attr = st.selectbox(
        "Select a sensitive attribute", sensitive_attr_list)
    if st.button("Submit"):

        # Group the dataset by quasi-identifier values
        grouped_data = df.groupby(
            quasi_identifiers).size().reset_index(name='count')

        # Identify small groups that violate the k-anonymity requirement
        small_groups = grouped_data[grouped_data['count'] < k]

        # Anonymize small groups using generalization
        for i, row in small_groups.iterrows():
            group_values = row[quasi_identifiers]
            generalized_values = []
            for qi in quasi_identifiers:
                # Generalize the quasi-identifier value to a higher level of abstraction
                generalized_value = generalize(group_values[qi], qi)
                generalized_values.append(generalized_value)
            # Replace the original quasi-identifier values with the generalized values
            df.loc[(df[quasi_identifiers] == group_values).all(axis=1), quasi_identifiers] = generalized_values

        df.to_csv("anonymized.csv", index=False)
        st.write(df)
        data = df.copy()
        anonymized_data = pd.read_csv("anonymized.csv")
        sensitive = sensitive_attr
        original_entropy = calculate_entropy(data[sensitive])
        anonymized_entropy = calculate_entropy(anonymized_data[sensitive])
        # create a bar chart showing the results
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(["Original", "Anonymized"], [original_entropy,anonymized_entropy], color=["blue", "orange"])
        ax.set_ylabel("Entropy")
        ax.set_title(
            "Entropy of Sensitive Attribute Before and After Anonymization")

        # show the results in Streamlit
        # st.pyplot(fig)
        # calculate the proportion of records with a unique quasi-identifier before and after anonymization

        original_proportion = calculate_proportion_unique(data, quasi_identifiers)
        anonymized_proportion = calculate_proportion_unique(anonymized_data, quasi_identifiers)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(["Original", "Anonymized"], [original_proportion,anonymized_proportion], color=["blue", "orange"])
        ax.set_ylabel("Proportion")
        ax.set_title(
            "Proportion of Records with Unique Quasi-Identifier Before and After Anonymization")

        # show the results in Streamlit
        # st.pyplot(fig)

        # Load the original dataset and the anonymized dataset
        original_data = df.copy()
        anonymized_data = pd.read_csv('anonymized.csv')
        print(len(original_data), len(anonymized_data))
        # Specify the quasi-identifier and sensitive attributes
        qi_attributes = quasi_identifiers
        # sensitive_attribute = 'occupation'
        sensitive_attribute = sensitive_attr
        # Calculate the information loss
        original_data_modes = original_data.groupby(qi_attributes)[sensitive_attribute].apply(lambda x: x.mode().iloc[0]).reset_index()
        anonymized_data_modes = anonymized_data.groupby(qi_attributes)[sensitive_attribute].apply(lambda x: x.mode().iloc[0]).reset_index()
        num_rows = min(len(original_data_modes), len(anonymized_data_modes))
        original_data_modes = original_data_modes.sample(num_rows, random_state=42).reset_index()
        anonymized_data_modes = anonymized_data_modes.sample(num_rows, random_state=42).reset_index()
        iloss = np.sum(original_data_modes !=anonymized_data_modes) / (num_rows)

        # Measure the degree of re-identification risk
        encoder = OneHotEncoder(sparse=False)
        encoded_data = encoder.fit_transform(anonymized_data[qi_attributes])
        distances = pairwise_distances(encoded_data, metric='hamming')
        risks = np.sum(distances < 1/(2*k), axis=1) / len(qi_attributes)

        # Define a function to create a scatter plot of the re-identification risk
        def create_scatter_plot(risks):
            data = pd.DataFrame({
                'Risk': risks,
                'Group': range(len(risks)),
            })
            chart = alt.Chart(data).mark_circle().encode(
                x='Group',
                y='Risk',
                size=alt.Size('Risk', scale=alt.Scale(range=[50, 500])),
                color=alt.Color('Risk', scale=alt.Scale(
                    scheme='redyellowgreen')),
            ).configure_axis(
                labelFontSize=20,
                titleFontSize=20,
            ).configure_text(
                fontSize=20,
            )
            return chart

        # Display the evaluation results
        st.write('# Anonymization Quality Evaluation')
        # st.write('## Information Loss')
        # st.write(iloss.drop('index'))
        st.write('## Re-identification Risk')
        st.write(create_scatter_plot(risks))

def pii_identify(df):
    # Define the regex pattern in a Presidio `Pattern` object:
    pan_number_pattern = Pattern(name="pan_number_pattern", regex="[A-Z]{5}[0-9]{4}[A-Z]{1}", score=0.5)

    # Define the recognizer with one or more patterns
    pan_number_recognizer = PatternRecognizer(supported_entity="PAN_NUMBER", patterns=[pan_number_pattern])

    # Define the regex pattern in a Presidio `Pattern` object:
    twitter_mention_pattern = Pattern(name="twitter_mention_pattern", regex="@(\\w+)", score=0.5)

    # Define the recognizer with one or more patterns
    twitter_mention_recognizer = PatternRecognizer(supported_entity="TWITTER_MENTION", patterns=[twitter_mention_pattern])

    # # Analyzer output
    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(pan_number_recognizer)
    analyzer.registry.add_recognizer(twitter_mention_recognizer)

    anonymized_tweets = []
    for tweet in df["Tweet Content"]:
        # print("original: ", tweet)
        analyzer_results = analyzer.analyze(text=tweet, language="en")
        for result in analyzer_results:
            tweet = tweet.replace(tweet[result.start : result.end], wordmap.returnString(tweet[result.start : result.end]))
            tweet = replace_with_synonym.replace_adjectives_with_synonyms(tweet)
        anonymized_tweets.append(tweet)
    df["Anonymized Tweet"] = anonymized_tweets
    df.drop(columns=["Tweet Content"], inplace=True)
    df.rename(columns={"Anonymized Tweet": "Tweet Content"}, inplace=True)
    st.write("PII identified tweets")
    st.write(df)

def laplace_mechanism(epsilon, sensitivity):
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale)
    return noise

def epsilon_differential_privacy(df, flag):
    selected_attributes=['Retweets Received','Likes Received','User Followers','User Following']    
    sensitivity = 1.0  
    epsilon = 0.1
    for column in selected_attributes:
        df[column] = round(df[column] + laplace_mechanism(epsilon, sensitivity))
        for i in range(len(df[column])):
            if df[column][i] < 0:
                df[column][i] = -1*df[column][i]

    st.write("Epsilon Differential Privacy:")
    st.write(df)

    if(flag):
        k_anonymity_algo(df)

st.title("Data Anonymizer")
uploaded_file = st.file_uploader("Upload dataset", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, nrows=50)
    st.write("Original dataset:")
    st.write(df)
    # Create the checkboxes
    epsilon_dp = st.checkbox("Epsilon Differential Privacy")
    k_anonymity = st.checkbox("K-Anonymity")

    if epsilon_dp and k_anonymity:
        pii_identify(df)
        epsilon_differential_privacy(df, True)

    if epsilon_dp and not k_anonymity:
        pii_identify(df)
        epsilon_differential_privacy(df, False)

    if k_anonymity and not epsilon_dp:
        pii_identify(df)
        k_anonymity_algo(df)

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

openai_api_key = "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"

# st.set_page_config(page_title="CSV Reader")
st.header("Upload anonymized CSV file")
user_csv = st.file_uploader("Upload your anonymized CSV file", type="csv")
if user_csv is not None:
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    agent = create_csv_agent(llm, user_csv, verbose=True)

    user_question = st.text_input("Enter your question")
    if user_question is not None and user_question!="":
        with st.spinner(text="In progress"):
            st.write(agent.run(user_question))


st.header("Image Anonymizer")

image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # OpenCV reads images in BGR format
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to fit the canvas size
    img = cv2.resize(img, (400, 400))

    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Select region to anonymize"):
        # Create a canvas with the uploaded image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        canvas = np.copy(img)

        # Create a rectangle selector using OpenCV
        r = cv2.selectROI(canvas)

        # Show the selected region with a red rectangle
        cv2.rectangle(canvas, (int(r[0]), int(r[1])), (int(r[0]+r[2]), int(r[1]+r[3])), (255, 0, 0), 2)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas = np.copy(canvas)
        st.image(canvas, caption="Region selected", use_column_width=True)

        # Apply Laplacian noise to the selected region
        epsilon = 0.005
        mask = np.zeros_like(img)
        mask[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 1
        noise = np.random.laplace(0, 1/epsilon, (400, 400, 3))
        anonymized = np.where(mask == 1, np.clip(img + noise, 0, 255).astype(np.uint8), img)

        anonymized = cv2.cvtColor(anonymized, cv2.COLOR_BGR2RGB)
        anonymized = np.copy(anonymized)
        st.image(anonymized, caption="Anonymized image", use_column_width=True)
