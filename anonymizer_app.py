import streamlit as st
import pandas as pd
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

st.title("Data Anonymizer")

# Load the Twitter data CSV file
uploaded_file = st.file_uploader("Upload Twitter data", type="csv")
if uploaded_file is not None:
    # Read the CSV file into a Pandas dataframe
    df = pd.read_csv(uploaded_file, nrows=10)
    # Show the original data table
    st.write("Original data table")
    st.write(df)

    # Define the regex pattern in a Presidio `Pattern` object:
    numbers_pattern = Pattern(name="numbers_pattern", regex="[A-Z]{5}[0-9]{4}[A-Z]{1}", score=0.5)

    # Define the recognizer with one or more patterns
    number_recognizer = PatternRecognizer(supported_entity="PAN_NUMBER", patterns=[numbers_pattern])

    # Analyzer output
    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(number_recognizer)

    # Anonymizer output
    anonymizer = AnonymizerEngine()

    # Define anonymization operators
    operators = {
        "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
        "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
        "PHONE_NUMBER": OperatorConfig(
            "mask",
            {
                "type": "mask",
                "masking_char": "*",
                "chars_to_mask": 7,
                "from_end": True,
            },
        ),
        "EMAIL_ADDRESS": OperatorConfig(
            "mask",
            {
                "chars_to_mask": 7,
                "type": "mask",
                "from_end": False,
                "masking_char": "*"
            }
        ),
        "TITLE": OperatorConfig("redact", {}),
    }

    # Create a new column "Anonymized Tweet" with the anonymized tweet
    anonymized_tweets = []
    for tweet in df["Tweet Content"]:
        analyzer_results = analyzer.analyze(text=tweet, language="en")
        anonymized_results = anonymizer.anonymize(
            text=tweet, analyzer_results=analyzer_results, operators=operators
        )
        anonymized_tweet = anonymized_results.text
        anonymized_tweets.append(anonymized_tweet)
    df["Anonymized Tweet"] = anonymized_tweets

    # Replace the "Tweet Content" column with the "Anonymized Tweet" column
    df.drop(columns=["Tweet Content"], inplace=True)
    df.rename(columns={"Anonymized Tweet": "Tweet Content"}, inplace=True)

    # Show the final data table with the anonymized tweets in the "Tweet Content" column
    st.write("Final data table with anonymized tweets")
    st.write(df)
