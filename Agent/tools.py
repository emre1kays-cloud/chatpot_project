from langchain.agents import Tool
from FinancialGoals.RAGToSQL.FabricsRAG import ask_fabric
import json
from datetime import datetime
from .memory import save_user_memory
from .memory import get_portfolio_from_memory

# Kullanıcının yatırım planını hesaplayan fonksiyon
def calculate_investment_plan(user_input: str) -> str:
    try:
        # Kullanıcı girişini JSON formatına dönüştürmeye çalış
        try:
            current_data = json.loads(user_input)  # Kullanıcıdan gelen veriyi JSON olarak ayrıştır
            client_id = current_data.get("client_id", "default_user")  # Kullanıcı ID'sini al, yoksa varsayılan bir ID ata
        except json.JSONDecodeError:
            # JSON ayrıştırma başarısız olursa, kullanıcı verisini temel bir sözlük olarak ele al
            current_data = {"goal_data": {"type": user_input}}
            client_id = "default_user"

        # Kullanıcının daha önce kaydedilmiş portföy verilerini al
        portfolio_data = get_portfolio_from_memory(client_id)

        # Kullanıcının mevcut verisi ile saklanan portföy verisini birleştir
        combined_data = {
            "portfolio_data": portfolio_data.get("portfolio_data", {}),  # Portföy verisi
            "goal_data": current_data.get("goal_data", {})  # Kullanıcının belirttiği yatırım hedefi
        }

        # Kullanıcının portföyündeki hisse senetleri ve hedefini al
        stocks = combined_data["portfolio_data"].get("stocks", "No stock data")
        goal = combined_data["goal_data"].get("type", "No goal specified")

        # Hata ayıklama amaçlı verileri ekrana yazdır
        print(f"DEBUG: Calculating investment plan with stored data - stocks: {stocks}, goal: {goal}")

        # Kullanıcıya önerilen yatırım planını JSON formatında döndür
        return json.dumps({
            "status": "success",
            "plan": {
                "stocks": stocks,
                "goal": goal,
                "recommended_investment": "$10,000/year"  # Örnek bir yatırım önerisi
            }
        })

    except Exception as e:
        # Hata oluşursa, hata mesajını içeren bir JSON döndür
        print(f"Error in calculate_investment_plan: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Error calculating plan: {str(e)}",
            "plan": None
        })

# Kullanıcıdan finansal analiz veya yatırım danışmanlığı isteyip istemediğini soran fonksiyon
def ask_initial_question(user_input: str) -> str:
    return "What are you looking for in terms of financial analysis or investment advice?"

# Kullanıcının hisse senedi portföyünü getiren fonksiyon
def get_stock_positioning(user_input: str) -> str:
    try:
        print("Starting get_stock_positioning...", user_input)
        client_id = user_input  # Kullanıcıdan gelen kimlik bilgisini al

        if user_input == "None":  # Eğer kullanıcı kimliği belirtilmemişse hata mesajı döndür
            print("user_input is null")
            return json.dumps({
                "status": "error",
                "message": "Please provide your client ID.",
                "portfolio_data": None
            })

        # Kullanıcının portföy verilerini veritabanından veya vektör deposundan al
        client_data = ask_fabric(
            f"Fetch all details for the client id {client_id} including their portfolios, assets, risk metrics, and recommended asset allocations."
        )

        # Alınan veriyi JSON formatında düzenle
        portfolio_data = {
            "status": "success",
            "portfolio_data": {
                "stocks": client_data,
                "last_updated": datetime.now().isoformat()  # En son güncelleme zamanı
            }
        }

        # Kullanıcının verilerini hafızaya kaydet
        save_user_memory(
            client_id,
            f"Fetching portfolio data for client {client_id}",
            "Portfolio data retrieved successfully",
            portfolio_data
        )

        return json.dumps(portfolio_data)  # JSON formatında veriyi döndür

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error: {str(e)}",
            "portfolio_data": None
        })

# Kullanıcıya finansal hedefini soran fonksiyon
def ask_financial_goal(user_input: str) -> str:
    try:
        # Kullanıcıdan gelen veriyi JSON formatında ayrıştır
        existing_data = json.loads(user_input) if user_input else {}

        return json.dumps({
            "status": "success",
            "message": "What are your specific financial goals?",
            "previous_data": existing_data  # Kullanıcının daha önce belirttiği hedefler
        })
    except:
        return "What are your specific financial goals? Please include target amount and timeline."

# Kullanıcının risk skorunu hesaplayan fonksiyon
def calculate_risk(user_input: str) -> str:
    return "Based on your portfolio, we calculate a risk score of 7/10 with moderate diversification metrics."

# Kullanıcının risk seviyesine göre bir yatırım planı öneren fonksiyon
def suggest_risk_plan(user_input: str) -> str:
    return "To mitigate risks, we suggest diversifying further into bonds and international equities."

# Kullanıcının yatırım hedeflerine uygun bir yatırım planı öneren fonksiyon
def suggest_investment_plan(user_input: str) -> str:
    return "Here’s a suggested investment plan tailored to your goal. Let us know if it meets your expectations."

# Kullanıcıdan geri bildirim alarak alternatif planlar sunan fonksiyon
def handle_feedback(user_input: str) -> str:
    if "alternative" in user_input.lower():
        return "Here’s an alternative plan: 40% stocks, 40% bonds, and 20% savings for a more conservative approach."
    return "Great! Let us know if you have any further questions or concerns."

# Kullanılacak tüm araçları (fonksiyonları) içeren bir liste döndüren fonksiyon
def get_tools():
    return [
        Tool(
            name="Ask Initial Question",
            func=ask_initial_question,
            description="Kullanıcının ihtiyaçlarını anlamak veya bağlamı belirlemek için konuşmayı başlatır."
        ),
        Tool(
            name="Get Stock Positioning",
            func=get_stock_positioning,
            description="Kullanıcının hisse senedi portföy verilerini getirir. Kullanıcı kimliği gerektirir."
        ),
        Tool(
            name="Ask Financial Goal",
            func=ask_financial_goal,
            description="Kullanıcının finansal hedeflerini (tasarruf, emeklilik vb.) sorgular."
        ),
        Tool(
            name="Calculate Risk",
            func=calculate_risk,
            description="Kullanıcının portföyündeki riskleri analiz eder ve bir risk skoru hesaplar."
        ),
        Tool(
            name="Suggest Risk Plan",
            func=suggest_risk_plan,
            description="Portföydeki riskleri azaltmak için önerilen risk yönetimi planını sağlar."
        ),
        Tool(
            name="Calculate Investment Plan",
            func=calculate_investment_plan,
            description="Kullanıcı girdisine ve portföy verilerine göre özel bir yatırım planı oluşturur."
        ),
        Tool(
            name="Suggest Investment Plan",
            func=suggest_investment_plan,
            description="Kullanıcının finansal hedeflerine uygun bir yatırım planı önerir."
        ),
        Tool(
            name="Handle Feedback",
            func=handle_feedback,
            description="Kullanıcıdan gelen geri bildirimlere yanıt vererek alternatif planlar sunar."
        ),
    ]
