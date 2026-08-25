import json
import os
import urllib.request
import boto3

# Initialize AWS Clients
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
sns_client = boto3.client(service_name="sns", region_name="us-east-1")

def query_tavily_search(bank_name):
    """Fetches highly recent search snippets and direct URLs via Tavily REST API."""
    url = "https://api.tavily.com/search"
    search_query = f'"{bank_name}" ("app down" OR "glitch" OR "outage" OR "fraud alert" OR "service failure")'
    
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": search_query,
        "search_depth": "advanced",
        "topic": "news",         # Focuses on active news coverage and incidents
        "days": 2,               # Strictly limits search to the last 48 hours
        "max_results": 5,
        "include_domains": ["news24.com", "fin24.com", "reddit.com", "x.com", "businesstech.co.za"]
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as resp:
            res_body = json.loads(resp.read().decode('utf-8'))
            results = res_body.get('results', [])
            
            context_text = "\n\n".join([f"Source: {r['url']}\nContent: {r['content']}" for r in results])
            
            # Filter out root/generic URLs
            valid_urls = []
            for r in results:
                raw_url = r.get('url', '')
                if raw_url and not any(bad in raw_url for bad in ['/r/all', '/downdetector', 'lang=en', 'status_is_down']):
                    valid_urls.append(raw_url)
            
            return context_text, valid_urls
    except Exception as e:
        print(f"Error querying Tavily API: {str(e)}")
        return "No web snippets retrieved due to search API error.", []

def send_email_alert(assessment, source_urls):
    """Sends a formatted email notification with direct links via Amazon SNS."""
    severity = assessment.get("highest_severity_score", 0)
    bank = assessment.get("bank", "Financial Institution")
    summary = assessment.get("summary", "No summary provided.")
    action = assessment.get("recommended_action", "N/A")
    top_complaints = assessment.get("top_complaints", [])

    complaint_lines = ""
    for c in top_complaints:
        complaint_lines += f"- [{c.get('category')}] (Severity {c.get('severity')}/5): {c.get('detail')}\n"

    url_lines = ""
    if source_urls:
        for idx, link in enumerate(source_urls, 1):
            url_lines += f"{idx}. {link}\n"
    else:
        url_lines = "No direct source URLs available."

    subject = f"🚨 HIGH RISK ALERT: {bank} (Severity {severity}/5)"
    
    email_body = f"""HIGH RISK SENTIMENT INCIDENT DETECTED

Institution: {bank}
Severity Level: {severity}/5

EXECUTIVE SUMMARY:
{summary}

TOP IDENTIFIED ISSUES:
{complaint_lines if complaint_lines else 'No specific categories flagged.'}

RECOMMENDED ACTION:
{action}

DIRECT POSTS & RECENT SOURCES FOUND:
{url_lines}
---
Generated automatically by AWS Bedrock Sentiment Agent.
"""

    try:
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=email_body
        )
        print("Email alert with verified direct sources dispatched successfully via SNS!")
    except Exception as e:
        print(f"Failed to send SNS alert: {str(e)}")

def run_sentiment_agent(bank_name="Standard Bank"):
    print(f"Searching web for sentiment regarding: {bank_name}...")
    
    # 1. Fetch search snippets and source URLs
    search_context, source_urls = query_tavily_search(bank_name)
    
    # 2. System Prompt
    system_prompt = """You are an Enterprise Risk & Sentiment Analysis Agent. 
Analyze the provided web/social snippets regarding a financial institution.
Extract risks, group complaints into themes, and assess severity on a 1-5 scale (1 = minor venting, 5 = widespread operational outage).

Return ONLY a valid JSON object matching this exact structure:
{
  "bank": "Bank Name",
  "overall_sentiment": "Negative | Neutral | Mixed",
  "highest_severity_score": 1-5,
  "incident_detected": true/false,
  "summary": "Brief 2-sentence executive summary",
  "top_complaints": [
     {"category": "Category Name", "detail": "Description", "severity": 1-5}
  ],
  "recommended_action": "Suggested next step"
}"""

    messages = [
        {"role": "user", "content": [{"text": f"Analyze these search snippets:\n\n{search_context}"}]}
    ]
    
    # 3. Call Bedrock
    try:
        response = bedrock_runtime.converse(
            modelId="us.amazon.nova-micro-v1:0",
            system=[{"text": system_prompt}],
            messages=messages,
            inferenceConfig={"temperature": 0.0}
        )
    except Exception as e:
        print(f"Bedrock Invocation Error: {str(e)}")
        return {"error": str(e)}
    
    # 4. Extract and Sanitize Output
    output_text = response['output']['message']['content'][0]['text']
    
    cleaned_text = output_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()

    try:
        assessment = json.loads(cleaned_text)
        severity = assessment.get("highest_severity_score", 0)
        
        print(f"Analysis complete. Severity score: {severity}/5")
        
        # Dispatch email if threshold is met
        if severity >= 2 and SNS_TOPIC_ARN:  # Set to >= 4 for production mode
            print("Alert condition met! Dispatched email alert...")
            send_email_alert(assessment, source_urls)
        else:
            print("Severity low or SNS ARN missing. Execution logged successfully.")
            
        return assessment

    except json.JSONDecodeError as e:
        print("Failed to parse JSON from model output:\n", output_text)
        return {"error": "JSON parse error", "raw": output_text}

def lambda_handler(event, context):
    result = run_sentiment_agent(bank_name="Standard Bank")
    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }