import sys, urllib.parse, urllib.request, re

def search(query, n=10):
    q = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={q}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='ignore')
    # extract result links
    links = re.findall(r'<a rel="nofollow" class="result__a" href="(.*?)"', html)
    titles = re.findall(r'<a rel="nofollow" class="result__a" href=".*?">(.*?)</a>', html)
    results = []
    for href, title in zip(links, titles):
        title = re.sub('<.*?>', '', title)
        results.append((title, href))
        if len(results) >= n:
            break
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ddg_scrape.py <query>")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    for title, link in search(query):
        print(f"{title}\n{link}\n")
