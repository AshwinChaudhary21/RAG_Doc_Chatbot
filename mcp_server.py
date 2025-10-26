from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import requests, datetime, json
from serpapi import GoogleSearch


load_dotenv()


app = Flask(__name__)

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

@app.route("/web_search", methods=["POST"])
def web_search():
    try:
        data = request.get_json()
        query = data.get("query")
        if not query:
            return jsonify({"error": "Query not provided"}), 400

        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY
        }

        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()

        results = response.json()
        snippets = []

        if "organic_results" in results:
            for i, item in enumerate(results["organic_results"][:5]):
                title = item.get("title", "No Title")
                snippet = item.get("snippet", "No Snippet")
                link = item.get("link", "#")
                snippets.append(f"Result {i+1}: {title}\n{snippet}\nLink: {link}\n")

        if snippets:
            return jsonify({"search_results": "\n".join(snippets)})
        else:
            return jsonify({"search_results": "No relevant web search results found."})

    except Exception as e:
        print(f"Internal Server Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500
    
if __name__ == '__main__':
    app.run(debug=True)