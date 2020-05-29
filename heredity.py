import csv
import itertools
import sys

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
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
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
    # overall joint probability of the whole dataset
    jointProb = 1

    for person, data in people.items():
        singleProb = 1
        father = data['father']
        mother = data['mother']
        if person in one_gene:
            if mother is None:
                singleProb *= PROBS["gene"][1]
            else:
                # given person has one gene, either got the gene from father or mother, not both
                # initialize probability of passing gene based on number of genes in parent
                # 2 genes = 0.99, 1 gene = 0.5, 0 genes = 0.01, with probability of mutation 0.01
                dadPass = PROBS['mutation']
                momPass = PROBS['mutation']
                if father in one_gene:
                    dadPass = 0.5
                elif father in two_genes:
                    dadPass = 1 - PROBS['mutation']
                if mother in one_gene:
                    momPass = 0.5
                elif mother in two_genes:
                    momPass = 1 - PROBS['mutation']

                singleProb *= (dadPass * (1-momPass) + (1-dadPass) * momPass)

            # joint probability of traits
            if person in have_trait:
                singleProb *= PROBS['trait'][1][True]
            else:
                singleProb *= PROBS['trait'][1][False]

        elif person in two_genes:
            if mother is None:
                singleProb *= PROBS['gene'][2]
            else:
                # given person has two genes, both parents passed to them
                # therefore the singleProb equation
                dadPass = PROBS['mutation']
                momPass = PROBS['mutation']
                if father in one_gene:
                    dadPass = 0.5
                elif father in two_genes:
                    dadPass = 1 - PROBS['mutation']
                if mother in one_gene:
                    momPass = 0.5
                elif mother in two_genes:
                    momPass = 1 - PROBS['mutation']

                singleProb *= (dadPass * momPass)
            
            if person in have_trait:
                singleProb *= PROBS['trait'][2][True]
            else:
                singleProb *= PROBS['trait'][2][False]

        else:
            if mother is None:
                singleProb *= PROBS['gene'][0]

            else:
                # given person has no genes, none of parents passed to them
                # therefore it is the negation of mom and dad pass
                dadPass = PROBS['mutation']
                momPass = PROBS['mutation']
                if father in one_gene:
                    dadPass = 0.5
                elif father in two_genes:
                    dadPass = 1 - PROBS['mutation']
                if mother in one_gene:
                    momPass = 0.5
                elif mother in two_genes:
                    momPass = 1 - PROBS['mutation']

                singleProb *= ((1-dadPass) * (1-momPass))
            
            if person in have_trait:
                singleProb *= PROBS['trait'][0][True]
            else:
                singleProb *= PROBS['trait'][0][False]

        jointProb *= singleProb

    return jointProb



def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person, data in probabilities.items():
        if person in one_gene:
            data['gene'][1] += p
        elif person in two_genes:
            data['gene'][2] += p
        else:
            data['gene'][0] += p
        
        if person in have_trait:
            data['trait'][True] += p
        else:
            data['trait'][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person, data in probabilities.items():
        sumGene = 0
        for i in range(3):
            sumGene += data['gene'][i]
        multiplierGene = 1/sumGene
        for i in range(3):
            data['gene'][i] = data['gene'][i] * multiplierGene

        sumTrait = data['trait'][True] + data['trait'][False]
        multiplierTrait = 1/sumTrait
        data['trait'][True] = data['trait'][True] * multiplierTrait
        data['trait'][False] = data['trait'][False] * multiplierTrait


if __name__ == "__main__":
    main()
