import sys
from app.rag import RAGPipeline


def main():
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question here"')
        return

    question = sys.argv[1]

    rag = RAGPipeline(file_path="data/sample.txt")
    result = rag.ask(question)

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source['id']} | score: {source['score']:.4f}")


if __name__ == "__main__":
    main()