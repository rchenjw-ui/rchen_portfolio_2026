import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    print(corpus)
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    links = corpus[page]
    #with probability 1-DAMPING FACTOR, return "random"
    #each link gets probability DAMPING FACTOR / num_links
    if len(links) > 0:
        link_probs = damping_factor / len(links)
        return_probs = {"random":1-damping_factor}
        for link in links:
            return_probs[link] = link_probs
    else:
        return_probs = {"random":1-damping_factor}
        for link in corpus:
            return_probs[link] = damping_factor / len(corpus)

    #handle zero links by making the sample link to everything
    return return_probs


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    #plan: start at the first page, and for n interations, should calculate transition model
    page = list(corpus.keys())[0]
    #note: for all sampling of the corpus keys need to list first

    out = {x:0 for x in corpus.keys()}
    for _ in range(0,n):
        #random sampling algorithm
        probs = transition_model(corpus, page, damping_factor)
        #given ordered dict, get some ranges
        rng = random.random()
        """
        #earlier page sampling algorithm, now use bucketing and dicts
        if rng < probs["random"]:
            page = random.choice(list(corpus.keys())) #randomly hops to a page
        else:
            choices = list(corpus[page]) #converts into list for random sampling
            page = random.choice(choices) #picks a page out of hyperlinks

        #i used the bucketing/dicts thing for utilizing a generalized prob distro
        """

        #how to handle zero links on page: if no links sample from all
        buckets = {}
        total = 0
        for c in probs:
            next_total = total + probs[c]
            buckets[c] = (total, next_total)
            total = next_total
        #now it should be in increasing order
        for link in buckets:
            if rng > buckets[link][0] and rng < buckets[link][1]:
                item = link
        if item == "random":
            choices = list(corpus[page]) #converts into list for random sampling
            if len(choices) == 0:
                item = random.choice(list(corpus.keys())) #chooses randomly from all if no pages
            else:
                item = random.choice(choices) #picks a page out of hyperlinks

        #iterate
        page = item

        #after that, increment the number of pages in out
        out[item] += 1


    #sum the values
    for page in out:
        out[page] = out[page]/n
    return out





def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    """
    specifications:
    The function accepts two arguments: corpus and damping_factor.
    The corpus is a Python dictionary mapping a page name to a set of all pages linked to by that page.
    The damping_factor is a floating point number representing the damping factor to be used in the PageRank formula.
    The return value of the function should be a Python dictionary with one key for each page in the corpus. Each key should be mapped to a value representing that page’s PageRank. The values in this dictionary should sum to 1.
    The function should begin by assigning each page a rank of 1 / N, where N is the total number of pages in the corpus.
    The function should then repeatedly calculate new rank values based on all of the current rank values, according to the PageRank formula in the “Background” section. (i.e., calculating a page’s PageRank based on the PageRanks of all pages that link to it).
    A page that has no links at all should be interpreted as having one link for every page in the corpus (including itself).
    This process should repeat until no PageRank value changes by more than 0.001 between the current rank values and the new rank values.
    """
    #convergence test: round everything to the nearest 10^-4, and if the delta is zero, return

    cont = True
    #generate equal odds
    links = list(corpus.keys())
    TOTAL = len(links)
    probs = {link:1/TOTAL for link in links}

    while cont:
        #print("iterated")
        next_probs = {}
        for link in links:
            #print(link)
            links_to = []
            #issue: need to find all pages that link to, not pages this page links to
            for i in corpus:
                #print(corpus[i])
                if link in corpus[i]:
                    links_to.append(i)

            #print(links_to)
            ranks_to = {i:probs[i] for i in links_to}
            num_links = {i:len(corpus[i]) for i in links_to}
            #print(ranks_to)
            #print(num_links)
            pageranks = [ranks_to[i]/num_links[i] for i in links_to]

            page_rank = (1-damping_factor)/TOTAL + damping_factor * sum(pageranks)
            #print(page_rank)

            next_probs[link] = page_rank

        #print("\n")
        #print(probs)
        #print(next_probs)
        #print("\n")

        #this is the termination constraint
        deltas = [abs(probs[l] - next_probs[l]) > 0.001 for l in links]
        probs = next_probs
        if any(deltas):
            cont = True
        else:
            cont = False
    return probs




    #each round of sampling returns a probability distribution



    #sampling a distribution: add up from bottom to top
    #note: since Python 3.7, dictionaries are ordered, so good!





if __name__ == "__main__":

    main()
