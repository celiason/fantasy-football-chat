# function to go from sql query to pandas dataframe

from sqlalchemy import text
from sqlalchemy import create_engine
import pandas as pd
import streamlit as st

def sql_to_df(query):

    # Connect to database
    password = st.secrets['supa_password']
    db_uri = f"postgresql://postgres.rpeohwliutyvtvmcwkwh:{password}@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
    engine = create_engine(db_uri, echo=True)
    db = engine.connect()

    # check i fquery is a file or a string
    if query.endswith(".sql"):
        query = open(query).read()

    # Execute query
    df = pd.read_sql_query(text(query), con=db)
    return df
