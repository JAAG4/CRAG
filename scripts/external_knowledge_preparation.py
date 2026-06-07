import argparse
import os
from tqdm import tqdm
import json
from utils import extract_keywords, select_relevants, rewrite_queries_tavily, TEST_SLICING

from phoenix.otel import using_metadata
from tracing_phx import tracer
import requests
from query_optimization import expand_query2doc, is_class_ii_query, decompose_query
from transformers import T5ForSequenceClassification, T5Tokenizer
from tavily import TavilyClient
from tavily.errors import BadRequestError
import os

tavily_client = TavilyClient(os.environ["TAVILY_KEY"])

UNTRUSTED_SOURCES_DOMAINS = [
    "naturalnews.com",
    "mercola.com",
    "healthimpactnews.com",
    "infowars.com",
    "newswars.com",
    "thegatewaypundit.com",
    "zerohedge.com",
    "breitbart.com",
    "rt.com",
    "sputniknews.com",
    "dailymail.co.uk",
    "nypost.com",
]
TRUSTED_SOURCES_DOMAINS = [
    "wikipedia.org",
    "nih.gov",
    "cdc.gov",
    "who.int",
    "mayoclinic.org",
    "thelancet.com",
    "nature.com",
    "science.org",
    "medlineplus.gov",
    "cochrane.org",
    "reuters.com",
    "apnews.com",
    "afp.com",
    "factcheck.org",
    "politifact.com",
    "snopes.com",
    "fullfact.org",
    "checkyourfact.com",
    "nytimes.com",
    "wsj.com",
    "bbc.com",
    "economist.com",
    "theguardian.com",
    "npr.org",
    "pbs.org",
]


@tracer.chain
def generate_knowledge_q(questions, task, openai_key, mode):
    """Query Rewriting step"""
    # if task == "bio":
    #     queries = [q[7:-1] for q in questions]
    # else:
    queries = extract_keywords(questions, task, openai_key)
    if mode == "wiki":
        search_queries = ["Wikipedia, " + e for e in queries]
    else:
        search_queries = queries
    return search_queries


@tracer.tool
def tavily_search_query_optimize(queries, output_file, tavily=tavily_client):
    with open(output_file, "w", encoding="utf-8") as outf:

        for query in tqdm(queries, desc="Searching for urls...", total=len(queries)):
            results_string = "#"
            with tracer.start_as_current_span(
                "tavily_search_query_optimize", openinference_span_kind="tool"
            ) as span:
                span.set_input({"query": query})
                if isinstance(query, dict):
                    query["topic"] = "general"
                    query["exclude_domains"] = UNTRUSTED_SOURCES_DOMAINS
                    query["include_domains"] = TRUSTED_SOURCES_DOMAINS
                    try:
                        tav_results = tavily.search(**query)
                    except BadRequestError as e:
                        print(f"\tError BadRequestError for query '{query}': {e}")
                elif isinstance(query, str) and query.strip() != "":
                    query = " ".join(
                        query.replace("claim:", "").replace("query:", "").split()
                    )
                    results_string = ""
                    if is_class_ii_query(query):
                        dec_queries = decompose_query(query)
                        print(f"-> Query '{query}'  Decomposed into: {dec_queries}")
                        expanded_queries = [
                            expand_query2doc(sq, retrieval_type="dense")
                            for sq in dec_queries
                        ]
                        for dq in expanded_queries:
                            try:

                                tav_results = tavily.search(
                                    query=dq,
                                    search_depth="advanced",
                                    max_results=5,
                                )
                                print(f"\tSub-query: `{dq}`")
                                all_responses_content = [
                                    tv_res["content"].replace("\n", " ")
                                    for tv_res in tav_results["results"]
                                ]
                                results_string += ". " + "; ".join(
                                    all_responses_content
                                )  # + "\n"
                            except BadRequestError as e:
                                print(
                                    f"\tError BadRequestError for sub-query '{dq}': {e}"
                                )
                            span.set_attributes({"sub_queries": dec_queries})
                        results_string = (
                            results_string.strip().strip(".").strip() + "\n"
                        )
                        span.set_output({"raw": tav_results, "processed": results_string})
                        # search_results.extend([{"queries":query,"results":rcontent}])
                        outf.write(results_string)
                    else:
                        try:
                            query = expand_query2doc(query, retrieval_type="dense")
                            tav_results = tavily.search(
                                query=query,
                                search_depth="advanced",
                                max_results=5,
                            )

                            print(f"\tQuery: `{query}`")
                            all_responses_content = [
                                tv_res["content"].replace("\n", " ")
                                for tv_res in tav_results["results"]
                            ]
                            results_string += "; ".join(all_responses_content) + "\n"
                            span.set_output({"raw": tav_results, "processed": results_string})
                            # search_results.extend([{"queries":query,"results":rcontent}])
                            outf.write(results_string)
                        except BadRequestError as e:
                            print(f"\tError BadRequestError for query '{query}': {e}")



@tracer.tool
def Search(queries, search_path, search_key):
    url = "https://google.serper.dev/search"
    responses = []
    search_results = []
    for query in tqdm(queries[:], desc="Searching for urls..."):
        payload = json.dumps({"q": query})
        headers = {"X-API-KEY": search_key, "Content-Type": "application/json"}

        reconnect = 0
        while reconnect < 3:
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                break
            except (requests.exceptions.RequestException, ValueError):
                reconnect += 1
                print("url: {} failed * {}".format(url, reconnect))
        result = json.loads(response.text)
        if "organic" in result:
            results = result["organic"][TEST_SLICING]
        else:
            results = query
        responses.append(results)

        search_dict = [{"queries": query, "results": results}]
        search_results.extend(search_dict)
    if search_path != "None":
        with open(search_path, "w") as f:
            output = json.dumps(search_results, indent=4)
            f.write(output)
    return search_results


@tracer.tool
def test_page_loader(url):
    from bs4 import BeautifulSoup
    import signal

    def handle(signum, frame):
        raise RuntimeError

    reconnect = 0
    while reconnect < 3:
        try:
            signal.signal(signal.SIGALRM, handle)
            signal.alarm(180)
            response = requests.get(url)
            break
        except (requests.exceptions.RequestException, ValueError, RuntimeError):
            reconnect += 1
            print("url: {} failed * {}".format(url, reconnect))
            if reconnect == 3:
                return []
    try:
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
    except:
        return []
    if soup.find("h1") is None or soup.find_all("p") is None:
        return []
    paras = []
    title = soup.find("h1").text
    paragraphs = soup.find_all("p")
    for _, p in enumerate(paragraphs):
        if len(p.text) > 10:
            paras.append(title + ": " + p.text)
    return paras


@tracer.tool
def visit_pages(questions, web_results, output_file, model_name, device, mode):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForSequenceClassification.from_pretrained(
        model_name, output_hidden_states=True
    )
    top_n = 5
    titles = []
    urls = []
    snippets = []
    queries = []
    for i, result in enumerate(web_results[:]):
        title = []
        url = []
        snippet = []
        if type(result["results"]) == list:
            for page in result["results"][:5]:
                if mode == "wiki":
                    if "wikipedia" in page["link"]:
                        title.append(page["title"])
                        url.append(page["link"])
                else:
                    title.append(page["title"])
                    url.append(page["link"])
                if "snippet" in page:
                    snippet.append(page["snippet"])
                else:
                    snippet.append(page["title"])
        else:
            titles.append([])
            urls.append([])
            snippets.append([result["results"]])
            queries.append(result["queries"])
            continue
        titles.append(title)
        urls.append(url)
        snippets.append(snippet)
        queries.append(result["queries"])
    output_results = []
    progress_bar = tqdm(range(len(questions[:])), desc="Visiting page content...")
    assert len(questions) == len(urls), (len(questions), len(urls))
    i = 0
    for title, url, snippet, query in zip(
        titles[i:], urls[i:], snippets[i:], queries[i:]
    ):
        if url == []:
            results = "; ".join(snippet)
        else:
            strips = []
            for u in url:
                strips += test_page_loader(u)
            if strips == []:
                output_results.append("; ".join(snippet))
                results = "; ".join(snippet)
            else:
                results, _ = select_relevants(
                    strips=strips,
                    query=questions[i],
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    top_n=top_n,
                )
        i += 1
        output_results.append(results.replace("\n", " "))
        progress_bar.update(1)
    with open(output_file, "w") as f:
        f.write("#")
        f.write("\n#".join(output_results))
    return output_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--input_queries", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--openai_key", type=str)
    parser.add_argument("--search_key", type=str)
    parser.add_argument(
        "--task", type=str, choices=["popqa", "pubqa", "arc_challenge", "bio"]
    )
    parser.add_argument("--search_path", type=str, default="None")
    parser.add_argument(
        "--mode",
        type=str,
        default="wiki",
        choices=["wiki", "all"],
        help="Optional strategies to modify search queries, wiki means web search engine tends to search from Wikipedia",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = args.openai_key
    with open(args.input_queries, "r") as query_f:
        questions = [q.strip() for q in query_f.readlines()][TEST_SLICING]
    with tracer.start_as_current_span(
        "generate_knowledge_q", openinference_span_kind="chain"
    ) as span:
        span.set_input(questions)
        # search_queries = generate_knowledge_q(
        #     questions, args.task, args.openai_key, args.mode
        # )
        # tavily_rewritten_queries = rewrite_queries_tavily(
        #     questions, args.task, args.openai_key
        # )

        span.set_attributes(
            {
                "task": args.task,
                "mode": args.mode,
                "queries": questions,
                # "output": tavily_rewritten_queries,
            }
        )

        # span.set_output(tavily_rewritten_queries)
    tavily_search_query_optimize(questions, f"{args.output_file}_qo_tavily.txt")


if __name__ == "__main__":
    main()
