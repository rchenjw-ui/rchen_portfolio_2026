import csv
import itertools
import sys
import random #delete at the end

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}

#can calculate base rate of condition is 0.0329

def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people) #only the keys, just the names of all people in people
    #print(powerset(names))
    for have_trait in powerset(names):


        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )

        if fails_evidence:
            continue #if fails_evidence, skip the set
        #conclusion: pass when have_trait only contain trait == True or Unknown



        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            #print(f"{one_gene} has one gene")
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    """
    This is the methods discussion. The goal is p(one_genes, two_genes, have_trait)
    Plan:
    - Start with gene probabilities, and given gene probabilities, calculate trait probabilities
    - Can check if any children of people have trait known or not
    - This is going to be a bit complicated, logistics:

    The requirement is to write three functions:
    - one function to calculate joint probabilities
    - One function for updating the probability distribution given new information P(A, B, C) one, two, have trait
    - One function to normalize all probabilities

    How to approach:
    - Note that suppose X depends on A, we can have P(x,a) = p(x|a)*p(a)
    - Thus, p(trait, gene) = p(trait|gene)*p(gene)
    - If uncorrelated, IE within same generation, just multiply
    - So, P(gene, child, parent) = p(child|gene, parent) * p(gene, parent)
    - P(gene, parent) = p(gene|parent) * p(parent)
    - Given that p(child|gene, parent) = p(child|gene) based on basics of bayes, final:
    - P(gene, child, parent) = p(child|gene, parent) * p(gene|parent) * p(parent)
    - P(trait, gene) = p(trait | gene) * p(gene)

    note: I can calculate probability of a person having a gene given both parent's distribution, or probability of trait given gene distribution
    What I am trying to figure out is, given traits of children and parents, what to do

    Note: could hypothetically do greedily, can iterate (ie, first calculate the probability of gene given trait, then reuse)
    This would add a complication, but IDK how I'd do this.



    So, total P(trait, gene, child, parents) = p(trait|gene) * p(child|gene) * p(gene|parents) * p(parents)

    The assumption is that because this is a bayesian network we only need to consider direct factors
    IE, p(trait|gene, parent) = p(trait|gene) because you already have gene

    For multi person scenario: P(parent, child) (this is the package deal from P(trait, gene, child, parent) from before)
    P(parent, child) = p(child|parent) * p(parent)

    Uncertain how I'd implement as algorithm though
    """

    """
    #find joint_prob = p(one, two, trait)
    #bit complicated, needs to do "all people in set"
    #Apparently, can just calculate all independently, p(a,b,c) = p(a)*p(b)*p(c)?
    #thus, more tractable, just iterate through all people in a set and calculate probability of one, two, and no genes, plus trait and no trait
    #define sets early:

    no_genes = set(people.keys()) - one_gene - two_genes #set subtraction
    no_trait = set(people.keys()) - have_trait
    probs_info = {
        person: {
            "gene": None,
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    #note: can just multiply as requested

    #need to get parents from person
    #may need to first calculate the parents furthest back, then children
    #use a dictionary for parents, just use default probabilities:

    #algorithm: DOES NOT WORK, TEST NEGATIVE
    #step 1 base rate the parents
    #step 2 set the probabilities given trait
    #step 3 layer by layer, calculate gene probability

    for c in probs_info:
        print(people[c])
        if probs_info[c]["gene"] == None:
            print(c)
            if people[c]["father"] == None: #sets parental to default
                probs_info[c]["gene"] = {
                    2:0.01,
                    1:0.03,
                    0:0.96
                }
            if people[c]["trait"] == 1: #sets priors given the trait
                probs_info[c]["gene"] = {
                    2:0.1976,
                    1:0.5106,
                    0:0.2918
                }
            if people[c]["trait"] == 0 and people[c]["trait"] != None:
                probs_info[c]["gene"] = {
                    2:0.0036,
                    1:0.0136,
                    0:0.9827
                }
            if people[c]["trait"] == None:
                probs_info[c]["gene"] = {
                    2: 0.01,
                    1: 0.03,
                    0: 0.96
                }

    transmit_odds = {
        0: 0.0001,
        1: 0.5,
        2: 0.99
    }

    for c in set(people.keys()):
        #given parents distribution calculate probability of one gene
        mother = people[c]["mother"]
        father = people[c]["father"]
        #transmission probs, probability (transmission) = 0.99[2] + 0.5[1] + 0.01^2 [0]
        mother_transmit = 0
        father_transmit = 0
        if mother != None and father != None:
            for i in range(0,3):
                mother_transmit += probs_info[mother]["gene"][i] * transmit_odds[i]
                father_transmit += probs_info[father]["gene"][i] * transmit_odds[i]
            two_probs = mother_transmit * father_transmit
            one_probs = mother_transmit + father_transmit - two_probs
            zero_probs = 1 - one_probs - two_probs
            probs_info[c]["gene"] = {
                2:two_probs,
                1:one_probs,
                0:zero_probs
            }

    #trait probabilities

    for c in probs_info:
        if people[c]["trait"] == 1:
            probs_info[c]["trait"][True] = 1
            probs_info[c]["trait"][False] = 0
        elif people[c]["trait"] == 0 and people[c]["trait"] != None:
            probs_info[c]["trait"][True] = 0
            probs_info[c]["trait"][False] = 1
        elif people[c]["trait"] == None:
            genes = probs_info[c]["gene"]
            trait_prob = genes[0] * 0.01+genes[1] * 0.56+genes[2] * 0.65

            probs_info[c]["trait"][True] = trait_prob
            probs_info[c]["trait"][False] = 1-trait_prob

    #one gene probability
    probability = 1
    for c in no_genes:
        probability = probability * probs_info[c]["gene"][1]
    for c in one_gene:
        probability = probability * probs_info[c]["gene"][1]
    for c in two_genes:
        probability = probability * probs_info[c]["gene"][2]
    for c in have_trait:
        probability = probability * probs_info[c]["trait"][True]
    for c in no_trait:
        probability = probability * probs_info[c]["trait"][False]

    print(probability)
    return probability
    #raise NotImplementedError
    """

    """
    no_genes = set(people.keys()) - one_gene - two_genes
    no_trait = set(people.keys()) - have_trait
    #set for no genes and set for no trait, for later joint probability multiplication

    probs_info = {
        person: {
            "gene": None,
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    for c in probs_info:
        #print(people[c])
        if probs_info[c]["gene"] == None:
            if people[c]["father"] == None: #sets parental to default
                probs_info[c]["gene"] = {
                    2:0.01,
                    1:0.03,
                    0:0.96
                }
            if people[c]["trait"] == 1: #sets priors given the trait
                probs_info[c]["gene"] = {
                    2:0.1976,
                    1:0.5106,
                    0:0.2918
                }
            if people[c]["trait"] == 0 and people[c]["trait"] != None:
                probs_info[c]["gene"] = {
                    2:0.0036,
                    1:0.0136,
                    0:0.9827
                }
            if people[c]["trait"] == None:
                probs_info[c]["gene"] = {
                    2: 0.01,
                    1: 0.03,
                    0: 0.96
                }

    transmit_odds = {
        0: 0.01,
        1: 0.5,
        2: 0.99
    }

    for c in set(people.keys()):
        #given parents distribution calculate probability of one gene
        mother = people[c]["mother"]
        father = people[c]["father"]

        #transmission probs, probability (transmission) = 0.99[2] + 0.5[1] + 0.01 [0]

        if mother != None and father != None:
            mother_transmit = 0
            father_transmit = 0
            for i in range(0,3):
                mother_transmit += probs_info[mother]["gene"][i] * transmit_odds[i]
                father_transmit += probs_info[father]["gene"][i] * transmit_odds[i]

            #generate odds
            two_probs = mother_transmit * father_transmit
            one_probs = mother_transmit + father_transmit - two_probs
            zero_probs = 1 - one_probs - two_probs
            probs_info[c]["gene"] = {
                2:two_probs,
                1:one_probs,
                0:zero_probs
            }

    #trait probabilities
    print(f"one gene: {one_gene}")
    print(f"two genes: {two_genes}")
    print(f"trait: {have_trait}")
    #print(probs_info)

    for c in probs_info:
        if people[c]["trait"] == 1:
            probs_info[c]["trait"][True] = 1
            probs_info[c]["trait"][False] = 0
        elif people[c]["trait"] == 0 and people[c]["trait"] != None:
            probs_info[c]["trait"][True] = 0
            probs_info[c]["trait"][False] = 1
        elif people[c]["trait"] == None:
            genes = probs_info[c]["gene"]
            trait_prob = 0
            for i in range(0,3):
                trait_prob = trait_prob + genes[i] * PROBS["trait"][i][True]
            probs_info[c]["trait"][True] = trait_prob
            probs_info[c]["trait"][False] = 1-trait_prob

    #joint probs

    #may need to do a little bit of dependency modelling
    #for trait and gene, P(trait AND gene) = p(trait|gene) * p(gene)
    #now, P(gene|trait) * p(gene|trait) for all, not P(gene) * p(trait) * p(gene) * p(trait)
    probability = 1
    for c in no_genes:
        probability = probability * probs_info[c]["gene"][0]
    for c in one_gene:
        probability = probability * probs_info[c]["gene"][1]
    for c in two_genes:
        probability = probability * probs_info[c]["gene"][2]
    for c in have_trait:
        probability = probability * probs_info[c]["trait"][True]
    for c in no_trait:
        probability = probability * probs_info[c]["trait"][False]

    print(probability)
    #note: probabilities seem anomalously small for joint probabilities, counting or not counting no_genes and no_trait
    return probability
    """

    """
    Final algorithm:
    This is just a joint probability, not a direct inference thing. So, we have all (gene, trait) pairs for all person, and we can just assume the probability of the whole thing
    We can calculate P(gene) either using background probability (for base) or parent information (IE you assume mother one and father two, what's here)
    Then, we calculate P(gene, trait) = P(trait|assumed genes) * P(assumed genes)

    We are making whole assumptions and then calculating the probability of the whole set of assumptions
    """
    def label_assignment(person):
            gene = ""
            if person in one_gene:
                gene = 1
            elif person in two_genes:
                gene = 2
            else:
                gene = 0

            trait = ""
            if person in have_trait:
                trait = True
            else:
                trait = False
            return {"trait":trait, "gene":gene}

    TRANSMIT_PROBS = {
        0:0.01,
        1:0.5,
        2:0.99
    }
    person_probs = {} #need to have structure: {person: {n_genes, trait, prob_n}}
    for person in people:
        #get parents
        mother = people[person]["mother"]
        father = people[person]["father"]

        #assign labels

        #after that, calculate transmission probs
        person_label = label_assignment(person)
        if mother is not None and father is not None:
            mother_label = label_assignment(mother)
            father_label = label_assignment(father)
            mother_gene = TRANSMIT_PROBS[mother_label["gene"]]
            father_gene = TRANSMIT_PROBS[father_label["gene"]]
            print(f"mother transmit odds: {mother_gene}")
            print(f"father transmit odds: {father_gene}")
            if person_label["gene"] == 0:
                #odds of zero:
                prob_n = 1- (mother_gene + father_gene)
            elif person_label["gene"] == 1:
                prob_n = mother_gene * (1-father_gene) + father_gene * (1-mother_gene)#probability mother transmits and father doesnt or vice versa
            else:
                prob_n = mother_gene*father_gene
        else:
            prob_n = PROBS["gene"][person_label["gene"]]

        #probs have been calculated for person, add to dict
        n_genes = person_label["gene"]
        trait = person_label["trait"]
        person_probs[person] = {"n_genes":n_genes, "trait":trait, "prob_n":prob_n}
    print(person_probs)
    joint_prob = 1
    for person in person_probs:
        person_data = person_probs[person]
        n_genes = person_data["n_genes"]
        trait = person_data["trait"]
        prob_n = person_data["prob_n"]
        trait_probs = PROBS["trait"][n_genes][trait] #p(trait|assumed genes)
        joint_prob = joint_prob * trait_probs * prob_n
    return joint_prob














def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    #P is P(one_gene, two_genes, have_trait)
    #P is a bit complicated, not sure how to udpate
    #update: given P(one_gene, two_genes, have_trait) = p(have_trait|one_gene, two_genes) * p(one_gene, two_genes)

    #not sure how to calculate either, but p(have_trait|one_gene, two_genes) could just be p(have_trait) overall if not in set?
    #p(one_genes, two_genes) = p(one_genes) * p(two_genes)
    #p = P(one_gene, two_genes, have_trait) = p(have_trait|one_gene, two_genes) * p(one_gene) * p(two_genes)

    #p(have_trait) may just be overall have_trait?

    for c in probabilities:
        if c in one_gene:
            probabilities[c]["gene"][1] = probabilities[c]["gene"][1]+p
        if c in two_genes:
            probabilities[c]["gene"][2] = probabilities[c]["gene"][2]+p
        if c not in one_gene and c not in two_genes:
            probabilities[c]["gene"][0] = probabilities[c]["gene"][0]+p
        if c in have_trait:
            probabilities[c]["trait"][True] = probabilities[c]["trait"][True]+p
        if c not in have_trait:
            probabilities[c]["trait"][False] = probabilities[c]["trait"][False]+p

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for c in probabilities: #keys, so person
        gene_probs = probabilities[c]["gene"]
        gene_total = gene_probs[2]+gene_probs[1]+gene_probs[0]
        for i in range(0,3):
            probabilities[c]["gene"][i] = probabilities[c]["gene"][i]/gene_total
        trait = probabilities[c]["trait"]
        total = trait[True]+trait[False]
        for i in [True,False]:
            probabilities[c]["trait"][i] = probabilities[c]["trait"][i]/total
    #test cases below
    #people = load_data("family2.csv")
    #probabilities = {
    #    person: {
    #        "gene": {
    #            2: random.random(),
    #            1: random.random(),
    #            0: random.random()
    #        },
    #        "trait": {
    #            True: random.random(),
    #            False: random.random()
    #        }
    #    }
    #    for person in people
    #}
    #for c in probabilities:
    #    print(c)
    #    print(probabilities[c]["gene"])
    #    print(probabilities[c]["trait"])
    #normalize(probabilities)
    #print("normalized")
    #for c in probabilities:
    #    print(c)
    #    print(probabilities[c]["gene"])
    #    print(probabilities[c]["trait"])


#NOTES:
"""
- "have_trait is just a hypothetical subset of people who might have the trait
 the code uses fails_evidence to skip impossible scenarios"

- "You can think of probability as how common that w is among all the cases where x y z are also true."

- " predict what genes someone has from the trait that they have or the other way around"

- "
"""

if __name__ == "__main__":
    people = load_data("family0.csv")
    p = joint_probability(people, {"Harry"}, {"James"}, {"James"})
    print(p)

    main()
