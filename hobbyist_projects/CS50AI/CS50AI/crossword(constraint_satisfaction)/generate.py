import sys

from crossword import *

import copy


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self): #done
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in self.domains:
            rm_sets = set()
            for val in self.domains[var]:
                if len(val) != var.length:
                    rm_sets.add(val)
            for val in rm_sets:
                self.domains[var].remove(val)

    def revise(self, x, y): #DONE
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """

        """
        check consistency:
        - call self.crossword.overlaps
        - using overlaps, find if there's an overlap, and for all iterations in X, see what Ys are possible
        - For x in self.domains x, for y in self.domains y, check at index of overlap

        How to check overlap: see index for both (given across or down + len)
        """
        overlaps = self.crossword.overlaps
        overlap = overlaps[(x,y)]
        remove = []
        if overlap != None:
            for x_val in self.domains[x]:
                consistent = False
                for y_val in self.domains[y]:
                    x_index = overlap[0]
                    y_index = overlap[1]
                    if x_val[x_index] == y_val[y_index]:
                        consistent = True #check if the two are consistent; ie same character at overlap
                if consistent == False:
                    remove.append(x_val)
            if remove:
                for val in remove:
                    self.domains[x].remove(val)
                return True
            return False
        else:
            return False


    def ac3(self, arcs=None): #DONE
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        """
        AC3: start with queue (FIFO), pick element zero and pop, run revise
        If revised, there will be values deleted/eliminated. If so, then add neighbors to domain

        Defining neighbors is the hard part
        """
        if arcs == None:
            arcs = [x for x in self.crossword.overlaps]
        while len(arcs) != 0: #makes sure queue is not zero
            item = arcs.pop(0)
            var0 = item[0]
            var1 = item[1]
            revised = self.revise(var0, var1)
            for var in self.domains:
                if len(self.domains[var]) == 0:
                    return False
            if revised:
                neighbors = []
                for arc in self.crossword.overlaps:
                    if var0 in arc and arc != item:
                        arcs = [arc] + arcs
        return True

        #neighbors def: find all connected to a variable in queue, add an arc
        #for each arc, run the revise algorithm and add all arcs
        #arcs contain variables connected to X (the first variable) that are not Y (the second)

    def assignment_complete(self, assignment): #done
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        total_variables = list(self.domains.keys())
        for var in total_variables:
            if var not in assignment:
                return False
        return True

    def consistent(self, assignment): #DONE
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        #check: run through self.crossword.overlaps, see indices, check if match
        #run through domains and such
        #check unary contraints too

        if len(assignment) == 1:
            return True
        for overlap in self.crossword.overlaps:
            #print(overlap)
            if overlap[0] in assignment and overlap[1] in assignment:
                var0 = assignment[overlap[0]]
                var1 = assignment[overlap[1]]
                #get assignment
                indices = self.crossword.overlaps[overlap]
                #print(indices)
                if indices is not None:
                    #print(indices[0])
                    letter0 = var0[indices[0]]
                    letter1 = var1[indices[1]]
                    if letter0 != letter1:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        raise NotImplementedError

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        #find unassigned variables by seeing assignment and total, possibly doing set ops
        assignment = set(assignment)
        total_vars = set([x for x in self.domains])
        unassigned = total_vars - assignment
        #Find domain and degree
        #domain, just use self.domains
        #create a dictionary
        len_dict = {}
        #print(unassigned)
        for var in unassigned:
            #print(var)
            domain = self.domains[var]
            len_dict[var] = len(domain)
        #find multiple max
        max_val = max(list(len_dict.values()))
        if list(len_dict.values()).count(max_val) > 1:
            #do the degrees check here
            overlaps = {}
            for pair in self.crossword.overlaps:
                for var in unassigned:
                    if var in pair:
                        if var not in overlaps:
                            overlaps[var] = 1
                        else:
                            overlaps[var] += 1
            max_val = max(list(overlaps.values()))
            for c in list(overlaps.keys()):
                if overlaps[c] == max_val:
                    return c
        for c in list(len_dict.keys()):
            if len_dict[c] == max_val:
                return c



        #degree, see the list for overlaps and see all overlaps including
        #otherwise, choose 0

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        #this is the challenge:

        #given assignment, check if complete
        if self.assignment_complete(assignment):
            return assignment
        #pick a variable
        var = self.select_unassigned_variable(assignment)

        #iterate through all possible values, possibly selecting a value given a heuristic
        domains = self.domains
        for val in domains[var]:
            new_assignment = copy.deepcopy(assignment)
            #Create a new assignment, selecting that value and doing other modifications if needed. Use deep copies if needed
            new_assignment[var] = val
            #print(new_assignment)
            #after selecting a value, optionally call the AC3 algorithm, but save that for later
            #check consistency of that value/variable assignment; could use unary or binary constraints, or both
            if self.consistent(new_assignment) == True:
                recursion = self.backtrack(new_assignment)
                if recursion != None:
                    return recursion
            #After that, recurse, the local copy of the assignment will then be passed down
            #If all fail, remove that value, then return failure
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()
    #print(assignment)

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":

    main()
