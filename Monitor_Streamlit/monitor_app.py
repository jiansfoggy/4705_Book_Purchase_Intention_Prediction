import boto3
import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.exceptions import EndpointConnectionError
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


DDB_TABLE_NAME = os.environ.get("DDB_TABLE", "Backend_Log_Cache")
DDB_REGION = os.environ.get("AWS_REGION", "us-east-1")


# ================================
# = Check Running Environment:   =
# = Local Machine or AWS Lab EC2 =
# ================================
def is_ec2_env():
    try:
        response = requests.get(
            "http://169.254.169.254/latest/meta-data/",
            timeout=0.2)
        return response.status_code == 200
    except requests.RequestException:
        return False


# ==============
# = Set up AWS =
# = DynamoDB   =
# ==============
# get temporary permit
def connect_dynamodb():
    # figure out the current environment: local or AWS learner lab EC2
    # connect to DynamoDB based on different environment
    if is_ec2_env():
        # boto3 will pick up credentials from env \
        # or ~/.aws/credentials automatically
        print("Detected EC2 (Learner Lab). Using IAM Role credentials...")
        dynamodb = boto3.resource("dynamodb", region_name=DDB_REGION)
        return dynamodb
    else:
        print("Detected local environment. \
               Using AWS credentials from env/config...")
        aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_token = os.getenv("AWS_SESSION_TOKEN")
        session = boto3.Session(
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
                aws_session_token=aws_token,
                region_name=DDB_REGION)
        # dynamodb = session.resource("dynamodb")
        # dynamodb = boto3.resource("dynamodb", region_name="us-east-1",
        #                           aws_access_key_id=aws_key,
        #                           aws_secret_access_key=aws_secret,
        #                           aws_session_token=aws_token)
        return session.resource("dynamodb")


def ensure_table(table_name=DDB_TABLE_NAME, create_if_missing=True,
                 wait_timeout=60):

    dynamodb = connect_dynamodb()
    table = dynamodb.Table(table_name)

    try:
        table.load()
        print(f"[DDB] Table '{table_name}' found.")
        return table
    except ClientError as e:
        err_code = e.response.get("Error", {}).get("Code", "")
        ex_ls2 = ["ResourceNotFoundException", "ValidationException"]
        if err_code not in ex_ls2:
            raise

        # load the existing DynamoDB table,
        # otherwise, create new DynamoDB table
        print(f"[DDB] Table '{table_name}' not found - creating...")
        er_ls = (ClientError, NoCredentialsError, EndpointConnectionError)
        try:
            new_table = dynamodb.create_table(
                    TableName=table_name,
                    AttributeDefinitions=[{"AttributeName": "text_hash",
                                           "AttributeType": "S"}],
                    KeySchema=[{"AttributeName": "text_hash", 
                                "KeyType": "HASH"}],
                    BillingMode="PAY_PER_REQUEST",
                    Tags=[{"Key": "final_project", "Value": "API_logs"}])
        except er_ls as create_err:
            print(f"[DDB] Failed to create table: {create_err}")
            raise

        t1 = new_table.meta.client.get_waiter('table_exists')
        d1 = {'Delay': 3, 'MaxAttempts': max(1, wait_timeout // 3)}
        t1.wait(TableName=table_name, WaiterConfig=d1)
        table = dynamodb.Table(table_name)
        print(f"[DDB] Created table {table_name}")
        return table


# =================
# = Get Variables =
# = From DynamoDB =
# =================
def log_dynamodb_caches1(table):
    """
    Scan the DynamoDB table and return three lists:
      texts: list of request_text
      preds: list of predicted_sentiment
      true_sent: list of true_sentiment (capitalized')
    """
    texts, preds, true_sent = [], [], []
    response = table.scan()
    items = response.get("Items", [])
    for it in items:
        texts.append(it.get("request_text", ""))
        preds.append(it.get("predicted_sentiment", "").capitalize())
        true_sent.append(it.get("true_sentiment", "").capitalize())
    return texts, preds, true_sent


def log_dynamodb_caches2(table=None):
    """
    Scan the DynamoDB table and return three lists:
      texts: list of request_text
      preds: list of predicted_sentiment
      true_sent: list of true_sentiment (capitalized')
    """
    expr_names = {"#r": "request_text",
                  "#p": "predicted_sentiment",
                  "#t": "true_sentiment"}
    projection = ", ".join(expr_names.keys())  # "#r, #p, #t"
    scan_kwargs = {
        "ProjectionExpression": projection,
        "ExpressionAttributeNames": expr_names,
    }
    texts, preds, true_sent = [], [], []
    try:
        resp = table.scan(**scan_kwargs)
        items = resp.get("Items", [])
        for it in items:
            texts.append(it.get("request_text", ""))
            preds.append(it.get("predicted_sentiment", "").capitalize())
            ts = it.get("true_sentiment", "")
            true_sent.append(ts.capitalize() if isinstance(ts, str) else ts)

        while "LastEvaluatedKey" in resp:
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
            resp = table.scan(**scan_kwargs)
            items = resp.get("Items", [])
            for it in items:
                texts.append(it.get("request_text", ""))
                preds.append(it.get("predicted_sentiment", "").capitalize())
                ts = it.get("true_sentiment", "")
                true_sent.append(
                    ts.capitalize() if isinstance(ts, str) else ts)

    except ClientError as e:
        try:
            st.error(f"[DDB] scan failed: {e}")
        except Exception:
            pass
        print(f"[DDB] scan ClientError: {e}")
        return [], [], []
    except Exception as e:
        try:
            st.error(f"[DDB] unexpected error during scan: {e}")
        except Exception:
            pass
        print(f"[DDB] unexpected scan error: {e}")
        return [], [], []

    return texts, preds, true_sent


# ======================
# = Load Movie Reviews =
# ======================
def log_imdb(path):
    if path.exists():
        imdb = pd.read_csv(path)
        st.write(f"Loaded {len(imdb)} IMDB reviews")
    else:
        st.error(f"IMDB dataset not found at {path} \
                   We customized dataset for further testing. \
                   Please use correct data file.")
        # st.stop()
        imdb = pd.DataFrame({"review": ["I loved it", "Bad movie"],
                             "sentiment": ["positive", "negative"]})
    return imdb


# ======================
# = Set Up Workflow    =
# = in main() function =
# ======================
def main():
    st.set_page_config(layout="wide")
    st.title("Monitor Dashboard -- Objective Movie \
              Review Sentiment Analyzer")
    st.text("This app monitors the running status of Objective \
             Movie Review Sentiment Analyzer, a FastAPI.")

    # 3. Load the Log and IMDB data
    st.header("1. Loading Log and IMDB Data")

    load_table = ensure_table(create_if_missing=True)
    print("Table status:", load_table.table_status)
    texts, preds, true_sent = log_dynamodb_caches2(table=load_table)

    st.write(f"Finish DataLoading. Loaded {len(texts)} log entries.")
    text_len = [len(t) for t in texts]
    text_len = sorted(text_len)
    st.write(f"Sample lengths from logs:\n{text_len[:5]}")
    st.write(f"Sample predictions from logs:\n{preds[:5]}")

    imdb_path = Path("./IMDB_Dataset.csv")
    imdb = log_imdb(imdb_path)

    imdb_len = [len(str(t)) for t in imdb["review"]]
    gts = imdb["sentiment"].tolist()
    st.write(f"Sample lengths from IMDB:\n{imdb_len[:5]}")
    st.write(f"Sample predictions from IMDB:\n{gts[:5]}")

    # 4. Compare the distribution of sentence lengths
    #    from both Log and IMDB data
    st.header("2. Data Drift Analysis -- Review \
               Lengths: IMDB vs. Log Requests")
    len_im = pd.DataFrame({
        "imdb_length": imdb_len
        })
    len_te = pd.DataFrame({
        "test_length": text_len
        })

    fig_imdb = px.histogram(
        len_im,
        x="imdb_length",
        nbins=25,
        histnorm="density",
        opacity=0.75,
        labels={"imdb_length": "Sentence Length"},
        title="IMDB Review Length"
        )

    lo = len_te['test_length'].min()
    hi = len_te['test_length'].max()
    bin_size = (hi - lo) / 25
    fig_test = px.histogram(
        len_te,
        x="test_length",
        # nbins=10,
        histnorm="density",
        opacity=0.75,
        labels={"test_length": "Sentence Length"},
        title="Logged Request Text Length"
        )

    fig_test.update_traces(xbins=dict(
        start=lo,
        end=hi,
        size=bin_size
    ))

    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("IMDB Review Length", "Request Text Length"),
        shared_yaxes=False
        )

    for trace in fig_imdb.data:
        fig1.add_trace(trace, row=1, col=1)

    for trace in fig_test.data:
        fig1.add_trace(trace, row=1, col=2)

    fig1.update_layout(height=450, width=800, showlegend=False,
                       title_text="Histograms of Sentence Lengths: \
                                   IMDB Review vs Logged Request Text",
                       template="plotly_white")

    st.plotly_chart(fig1, use_container_width=True)

    # 5. Bar chart of sentiment distributions
    st.header("3. Target Drift Analysis -- Sentiment \
               Distribution: IMDB vs. Log Requests")
    # IMDB dataset has a 'sentiment' column with values 'positive'/'negative'
    imdb_counts = imdb["sentiment"].value_counts().reset_index()
    imdb_counts.columns = ["sentiment", "count"]
    imdb_counts["source"] = "IMDB"

    log_counts = pd.Series(preds).value_counts().reset_index()
    log_counts.columns = ["sentiment", "count"]
    log_counts["source"] = "Logs"

    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("IMDB Sentiment Counts", "Logged Sentiment Counts"),
        shared_yaxes=False
        )

    # Add IMDB bar chart in column 1
    fig2.add_trace(
        go.Bar(
            x=imdb_counts["sentiment"],
            y=imdb_counts["count"],
            name="IMDB"
            ),
        row=1, col=1
        )

    # Add Logs bar chart in column 2
    fig2.add_trace(
        go.Bar(
            x=log_counts["sentiment"],
            y=log_counts["count"],
            name="Logs",
            marker_color='orange'
            ),
        row=1, col=2
        )

    fig2.update_layout(
        title="IMDB vs. Logs Sentiment Counts",
        showlegend=True, width=800, height=400,
        template="plotly_white"
        )

    fig2.update_yaxes(title_text="Count", row=1, col=1)
    fig2.update_yaxes(title_text="Count", row=1, col=2)

    # bar_df = pd.concat([imdb_counts, log_counts], ignore_index=True)
    # fig2 = px.bar(
    #     bar_df,
    #     x="sentiment",
    #     y="count",
    #     color="source",
    #     barmode="group",
    #     title="IMDB vs. Logs Sentiment Counts"
    #     )
    # # disable shared y-axis so each subplot scales independently
    # fig2.update_yaxes(matches=None)
    st.plotly_chart(fig2, use_container_width=True)

    # 6. Model Accuracy & User Feedback:
    st.header("4. Model Accuracy & User Feedback -- Compute \
               the Accuracy and Precision for Log Requests")
    st.write(f"{preds},\n{true_sent}.")
    accuracy = accuracy_score(true_sent, preds)
    precision = precision_score(true_sent, preds, average="macro",
                                zero_division=0)

    if accuracy < 0.80:
        st.error(f"Model accuracy {accuracy:.2%} already drops below 80%: ")

    st.metric("Accuracy", f"{accuracy:.2%}")
    st.metric("Precision (macro)", f"{precision:.2%}")


if __name__ == "__main__":
    main()
