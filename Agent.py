import os
import requests
import openai

# Retrieve API keys from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
news_api_key = os.environ.get("NEWS_API_KEY")

if not openai_api_key:
    raise ValueError("The environment variable OPENAI_API_KEY is not set.")
if not news_api_key:
    raise ValueError("The environment variable NEWS_API_KEY is not set.")

# Set the OpenAI API key for the openai library
openai.api_key = openai_api_key


# Base Agent class
class Agent:
    def __init__(self, name):
        self.name = name

    def perceive(self, input_data):
        """Receive input from the environment."""
        raise NotImplementedError("Perceive method must be implemented.")

    def decide(self):
        """Make decisions based on perceived input."""
        raise NotImplementedError("Decide method must be implemented.")

    def act(self):
        """Perform an action based on the decision."""
        raise NotImplementedError("Act method must be implemented.")


# Input Agent: Accepts the research topic.
class InputAgent(Agent):
    def __init__(self, name):
        super().__init__(name)
        self.topic = None

    def perceive(self, input_data):
        """Receive the research topic."""
        self.topic = input_data

    def decide(self):
        """Decide what to do with the topic."""
        return f"Proceeding with research on: {self.topic}"

    def act(self):
        """Print decision and return topic."""
        decision = self.decide()
        print(decision)
        return self.topic


# Retrieval Agent: Fetches articles from a news API.
class RetrievalAgent(Agent):
    def __init__(self, name, api_url, api_key):
        super().__init__(name)
        self.api_url = api_url
        self.api_key = api_key
        self.topic = None

    def perceive(self, topic):
        """Receive the topic to search for articles."""
        self.topic = topic

    def decide(self):
        """Make an API request based on the topic."""
        query_params = {"q": self.topic, "apiKey": self.api_key}
        response = requests.get(self.api_url, params=query_params)
        return response

    def act(self):
        """Fetch and return articles if available."""
        response = self.decide()
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            print(f"Retrieved {len(articles)} articles.")
            return articles
        else:
            print("Failed to retrieve articles.")
            return []


# Summarization Agent: Summarizes the content using OpenAI API.
class SummarizationAgent(Agent):
    def __init__(self, name):
        super().__init__(name)
        self.articles = None

    def perceive(self, articles):
        """Receive articles to summarize."""
        self.articles = articles

    def decide(self):
        """Summarize each article's content using the OpenAI API."""
        summaries = []
        for article in self.articles:
            content = article.get("content", "")
            if not content:
                summaries.append("No content to summarize.")
                continue

            prompt = f"Summarize the following article:\n\n{content}"
            try:
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    max_tokens=100
                )
                summary = response.choices[0].text.strip()
                summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing article: {e}")
                summaries.append("Summary unavailable.")
        return summaries

    def act(self):
        """Print and return summaries."""
        summaries = self.decide()
        for idx, summary in enumerate(summaries):
            print(f"Summary {idx + 1}: {summary}")
        return summaries


# File Storage Agent: Saves summaries to a file.
class FileStorageAgent(Agent):
    def __init__(self, name):
        super().__init__(name)
        self.summaries = None

    def perceive(self, summaries):
        """Receive summaries to store."""
        self.summaries = summaries

    def decide(self):
        """Indicate the action performed."""
        return "Summaries saved to research_summaries.txt."

    def act(self):
        """Write summaries to a file."""
        with open("research_summaries.txt", "w", encoding="utf-8") as file:
            for summary in self.summaries:
                file.write(summary + "\n\n")
        message = self.decide()
        print(message)
        return message


# Workflow class to orchestrate the agents
class Workflow:
    def __init__(self, agents):
        self.agents = agents

    def run(self, input_data):
        current_data = input_data
        for agent in self.agents:
            agent.perceive(current_data)
            current_data = agent.act()
        print("Workflow completed.")


if __name__ == "__main__":
    # Define the API URL for article retrieval (using the news API)
    NEWS_API_URL = "https://newsapi.org/v2/everything"

    # Instantiate agents
    input_agent = InputAgent(name="InputAgent")
    retrieval_agent = RetrievalAgent(name="RetrievalAgent", api_url=NEWS_API_URL, api_key=news_api_key)
    summarization_agent = SummarizationAgent(name="SummarizationAgent")
    file_storage_agent = FileStorageAgent(name="FileStorageAgent")

    # Set up the workflow: input -> retrieval -> summarization -> file storage
    agents = [input_agent, retrieval_agent, summarization_agent, file_storage_agent]
    research_workflow = Workflow(agents)

    # Run the workflow with a research topic
    topic = "AI in Healthcare"
    research_workflow.run(topic)
