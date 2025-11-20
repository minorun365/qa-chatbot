"""
QA Chatbot with Strands Agents and Streamlit

このアプリケーションは、Strands Agentsを使用したQAチャットボットです。
- Tavily検索（リモートMCPサーバー経由）
- Bedrockナレッジベースからの過去ナレッジ検索
を組み合わせて、ユーザーの質問に回答します。
"""

import os
import asyncio
import streamlit as st
import boto3
from strands import Agent
from strands.models import BedrockModel
from strands.tools import tool
from strands_tools import retrieve
from tavily import TavilyClient


# ===== 環境変数の設定 =====
# Streamlit Cloud と ローカル開発環境で環境変数を切り替える
def get_env_variable(key: str, default: str = None) -> str:
    """
    環境変数を取得する。Streamlit Cloudではst.secretsから、
    ローカルではシークレットTOMLファイルまたは環境変数から取得。
    """
    # Streamlit Cloudの場合: st.secretsから取得
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    
    # ローカル開発の場合: 環境変数から取得
    return os.getenv(key, default)


# ===== AWS認証情報の設定 =====
AWS_REGION = get_env_variable("AWS_REGION", "us-west-2")  # オレゴンリージョン
AWS_ACCESS_KEY_ID = get_env_variable("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = get_env_variable("AWS_SECRET_ACCESS_KEY")
KNOWLEDGE_BASE_ID = get_env_variable("KNOWLEDGE_BASE_ID")
TAVILY_API_KEY = get_env_variable("TAVILY_API_KEY")


# ===== Boto3セッションの作成 =====
def create_boto_session():
    """AWS Boto3セッションを作成"""
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        return boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
    else:
        # 認証情報がない場合はデフォルトの認証を使用
        return boto3.Session(region_name=AWS_REGION)


# ===== ツールの定義 =====
@tool
def tavily_search(query: str) -> str:
    """
    Web検索を行い、最新の情報を取得します。

    Args:
        query: 検索クエリ

    Returns:
        検索結果のテキスト
    """
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    results = tavily_client.search(query)
    return str(results)


# ===== エージェントの初期化 =====
def initialize_agent():
    """
    Strands Agentを初期化する。

    Returns:
        Agent: 初期化されたエージェント
    """
    # Boto3セッションを作成
    boto_session = create_boto_session()

    # BedrockModelの設定（Claude Sonnet 4.5 USクロスリージョン）
    bedrock_model = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        boto_session=boto_session,
        streaming=True,
    )

    # Agentの作成
    agent = Agent(
        model=bedrock_model,
        tools=[
            retrieve,       # Bedrockナレッジベース検索ツール
            tavily_search   # Tavily検索ツール（Python SDK経由）
        ],
        system_prompt="""あなたは親切なQAアシスタントです。
AIアプリ開発講座でチューターの代わりに受講者からの質問に答えてください。
先週が初回（AIアプリ開発入門講座＋APIハンズオン）で、今回は2回目（RAG構築入門）です。
簡単な講義のあと、前回使ったAWSアカウント ＋ GitHub Codespacesを継続利用してハンズオンします。

ユーザーからの質問に対して、以下のツールを活用して回答してください：
1. retrieve: 過去のナレッジベースから関連情報を検索します。まず最初にこれを使って過去の情報を確認してください。
   - 必ずknowledgeBaseId="{kb_id}"とregion="{region}"を指定してください。
2. tavily_search: Web検索を行い、最新の情報を取得します。過去ナレッジで見つからない場合や、最新情報が必要な場合に使用してください。

回答の際は、以下の点に注意してください：
- 日本語で回答してください
- ユーザーはIT初心者です。平易な言葉で、情報量が多くなりすぎないよう答えてください
- 原則retrieveのみを使い、どうしても策がない場合のみtavilyを使ってください
- 情報源を明示してください（過去ナレッジからか、Web検索からか）
- 正確で分かりやすい説明を心がけてください
- 不明な点がある場合は、正直に「分かりません」と伝えてください
- このチャットボットは簡易対応用なので、すぐに解決しない場合はチューターを呼ぶように伝えてください
""".format(kb_id=KNOWLEDGE_BASE_ID, region=AWS_REGION)
    )

    return agent


# ===== Streamlit UIの設定 =====
st.set_page_config(
    page_title="お助けチャットボット",
    page_icon="💪",
)

st.title("ハンズオンお助けチャットボット")
st.markdown("Strands Agentがチューターの代わりに答えてくれます。機密情報は入れないでね！")

# ===== チャット履歴の初期化 =====
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== チャット履歴の表示 =====
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ===== エージェントの初期化 =====
if 'agent' not in st.session_state:
    try:
        st.session_state.agent = initialize_agent()
    except Exception as e:
        st.error(f"エージェントの初期化に失敗しました: {str(e)}")
        st.stop()

agent = st.session_state.agent


# ===== ユーザー入力の処理 =====
if prompt := st.chat_input("質問を入力してください"):
    # ユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # アシスタントの応答を生成
    with st.chat_message("assistant"):
        container = st.container()

        try:
            # ストリーミング用のヘルパー関数
            def extract_tool_info(chunk):
                """チャンクからツール情報を抽出"""
                event = chunk.get('event', {})
                if 'contentBlockStart' in event:
                    tool_use = event['contentBlockStart'].get('start', {}).get('toolUse', {})
                    return tool_use.get('toolUseId'), tool_use.get('name')
                return None, None

            def extract_text(chunk):
                """チャンクからテキストを抽出"""
                if text := chunk.get('data'):
                    return text
                elif delta := chunk.get('delta', {}).get('text'):
                    return delta
                return ""

            async def stream_response():
                """レスポンスをストリーミング表示"""
                text_holder = container.empty()
                buffer = ""
                shown_tools = set()

                async for chunk in agent.stream_async(prompt):
                    if isinstance(chunk, dict):
                        # ツール実行を検出して表示
                        tool_id, tool_name = extract_tool_info(chunk)
                        if tool_id and tool_name and tool_id not in shown_tools:
                            shown_tools.add(tool_id)
                            # 現在のテキストを表示してから、ツールステータスを表示
                            if buffer:
                                text_holder.markdown(buffer)
                                buffer = ""
                            container.info(f"🔧 **{tool_name}** ツールを実行中...")
                            text_holder = container.empty()

                        # テキストを抽出して表示
                        if text := extract_text(chunk):
                            buffer += text
                            text_holder.markdown(buffer + "▌")

                # 最終表示（カーソル削除）
                if buffer:
                    text_holder.markdown(buffer)

                return buffer

            # 非同期実行
            loop = asyncio.new_event_loop()
            full_response = loop.run_until_complete(stream_response())
            loop.close()

            # アシスタントメッセージを履歴に追加
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            import traceback
            st.code(traceback.format_exc())