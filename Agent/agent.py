# Gerekli kütüphanelerin ve modüllerin import edilmesi
from langchain.agents import initialize_agent, AgentType  # LangChain ajanını başlatmak ve türünü belirlemek için kullanılır
from langchain_community.chat_models import ChatOpenAI  # OpenAI'nin sohbet modeliyle entegrasyon sağlar
from .tools import get_tools  # Kullanılacak araçları (tools) içe aktaran özel bir modül
from .memory import load_user_memory  # Kullanıcının hafızasını yüklemek için özel bir modül
from langchain.agents import AgentOutputParser  # Ajanın çıktısını ayrıştıran (parse eden) sınıf
from langchain.schema import AgentAction, AgentFinish  # Ajanın alabileceği olası çıktıları temsil eden sınıflar
import re  # Düzenli ifadeler (regex) ile metin işlemleri yapmak için
from typing import Union  # Birden fazla veri türü döndürebilen fonksiyonlar için
import json  # JSON formatında veri işlemleri yapmak için

# Özel bir çıktı ayrıştırıcı (parser) sınıfı oluşturuluyor
class JSONOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        Bu fonksiyon, LLM (Language Model) tarafından üretilen çıktıyı ayrıştırır (parse eder).
        Eğer çıktı bir aksiyon (Action) içeriyorsa, onu işler.
        Aksi halde işlemi tamamlanmış olarak işaretler (AgentFinish döndürür).
        """
        try:
            # LLM çıktısından "Action" ve "Action Input" değerlerini almak için düzenli ifadeler kullanılıyor
            action_match = re.search(r'Action: (.*?)[\n]', llm_output, re.DOTALL)
            action_input_match = re.search(r'Action Input: (.*?)[\n]', llm_output, re.DOTALL)

            if action_match and action_input_match:
                action = action_match.group(1).strip()  # Eylem (Action) bilgisini al ve boşlukları temizle
                action_input = action_input_match.group(1).strip()  # Eylem girdisini (Action Input) al ve temizle

                # Eğer belirli aksiyonlar için JSON formatı zorunluysa, JSON doğrulaması yap
                if action in ["Calculate Investment Plan", "Get Stock Positioning"]:
                    try:
                        # Eğer JSON geçerli değilse, bir hata fırlatacak
                        json.loads(action_input)
                    except json.JSONDecodeError:
                        # Geçersiz JSON tespit edilirse, input'u JSON formatına uygun bir yapıya çevir
                        action_input = json.dumps({"user_input": action_input})

                return AgentAction(action, action_input, llm_output)  # Ajanın aksiyon almasını sağla

            # Eğer bir aksiyon tespit edilmezse, işlem tamamlanmış olarak işaretlenir
            return AgentFinish(
                return_values={"output": llm_output},  # Kullanıcıya çıktıyı döndür
                log=llm_output,  # Log kaydına ekle
            )
        except Exception as e:
            # Bir hata oluşursa, hata mesajını içeren bir AgentFinish nesnesi döndür
            return AgentFinish(
                return_values={"output": f"Error parsing output: {str(e)}"},
                log=llm_output,
            )

# Kullanıcıya özel bir ajan oluşturma fonksiyonu
def create_agent(user_id):
    """
    Bu fonksiyon, verilen kullanıcı kimliğine (user_id) göre bir LangChain ajanı oluşturur.
    Ajan, finansal danışmanlık alanında çalışacak şekilde yapılandırılmıştır.
    """

    # OpenAI'nin GPT-4o-mini modelini düşük sıcaklıkla (daha az rastgele) kullan
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # Kullanılacak araçları (tools) getir
    tools = get_tools()

    # Ajan için özel bir istem (prompt) şablonu tanımla
    PROMPT_TEMPLATE = """
    Sen zeki bir finans danışmanı asistansın. Kullanıcıyla etkileşimlerinde bağlamı koruyarak verileri JSON formatında araçlara iletmelisin.

    KRİTİK İŞ AKIŞI:
    1. Her zaman önce "Get Stock Positioning" aracını çağır.
    2. Bu aracın JSON formatındaki çıktısını ayrıştır (parse et).
    3. Kullanıcının hedefleriyle birlikte portföy verisini kullanarak diğer araçları çağır.

    Veri İşleme Kuralları:
    - Tüm araç girdileri ve çıktıları JSON formatında olmalıdır.
    - Araç çıktıları ayrıştırılmalı (parse edilmeli) ve doğrulanmalıdır.
    - Yeni bir araç çağırırken, önceki aracın verilerini de eklemelisin.

    Örnek Doğru İş Akışı:
    Kullanıcı: "Ev almak istiyorum"
    Düşünce: Önce hisse senedi pozisyonunu öğrenmeliyim
    Aksiyon: Get Stock Positioning
    Aksiyon Girdisi: new_user
    Gözlem: {{"status": "success", "portfolio_data": {{"stocks": "AAPL: 100, GOOGL: 50"}}}}
    Düşünce: Şimdi portföy verisine sahibim, yatırım planı hesaplayabilirim
    Aksiyon: Calculate Investment Plan
    Aksiyon Girdisi: {{"portfolio_data": {{"stocks": "AAPL: 100, GOOGL: 50"}}, "goal_data": {{"type": "house", "timeline": "5 years", "target_amount": "500000"}}}}

    Önemli Noktalar:
    - TÜM araç etkileşimleri JSON formatında olmalıdır.
    - Sonraki araç çağrılarına önceki araç verilerini eklemelisin.
    - Araç yanıtlarını DOĞRULA ve ayrıştır.

    Hatalı veya Eksik Veri İşleme:
    - Eğer "Get Stock Positioning" başarısız dönerse, kullanıcıdan ek bilgi iste.
    - Eğer veri eksikse, hesaplama yapmadan önce eksik bilgileri topla.
    - Araç yanıtlarının "status" alanını her zaman kontrol et.

    {tool_descriptions}
    """

    # Araç açıklamalarını PROMPT'e ekleyerek formatı tamamla
    prompt = PROMPT_TEMPLATE.format(
        tool_descriptions="\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    )

    # Kullanıcının hafızasını yükle (geçmiş konuşmaları ve bağlamı saklamak için)
    memory, vectorstore = load_user_memory(user_id)

    # Ajanı başlat
    agent = initialize_agent(
        tools=tools,  # Araçları (tools) ajan ile entegre et
        llm=llm,  # OpenAI LLM modelini kullan
        agent="conversational-react-description",  # Konuşmaya dayalı bir ReAct ajanı türü
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Önceden eğitilmemiş, anlık karar veren bir ajan türü
        memory=memory,  # Kullanıcının önceki konuşmalarını hatırlaması için hafıza ekle
        verbose=True,  # Konsolda detaylı log çıktılarını göster
        prompt=prompt,  # Önceden tanımlanan PROMPT_TEMPLATE kullan
        handle_parsing_errors=True  # Ayrıştırma hatalarını yöneterek, sistemin çökmesini engelle
    )

    return agent  # Ajanı döndür
