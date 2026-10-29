import itertools
import random
import time #this is for testing, remove before flight
"""
algorithm:
- you know safe from successful clicks; mark that cell as safe and add to explored
- Add the sentence and count
- To mark a cell as safe in a sentence, remove the cell from the sentence while keeping count constant
- Then, will need to run pairwise comparison algorithm,
- After pairs, for all with diff len, create Sentence(cells(longer) - cells(shorter), count(longer)-count(shorter))
- Then, for these successful pairs, delete the original Sentences
- Repeat a few times for all the pairs you can get
- At the end, there will be a few Sentences with single cell and count 1, mark these as mines

Second part of algorithm:
- If there are with overlap, you add them, remove duplicates (sets do that) and then count = total-overlap


so far, as of 3/18/2026, spent a total of 2-3 hours on this

NOTES:
Remember
AND of two sets of cells with both N=1 has count = 1
XOR of any set of cell with 1 = N-1 count

"""


#issues to work on: empty cells,
#did not infer all possible safe cells

#note I is row, J is column

class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8): #DONE

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly, i is height j is width
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self): #DONE
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell): #DONE
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell): #DONE
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self): #DONE
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count): #DONE
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other): #DONE
        return self.cells == other.cells and self.count == other.count

    def __str__(self): #DONE
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        out_set = set()

        if self.count == len(self.cells):
            for c in self.cells:
                out_set.add(c)
        return out_set

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        out_set = set()
        if self.count == 0:
            for c in self.cells:
                out_set.add(c)
        return out_set

    def mark_mine(self, cell): #DONE
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count = self.count-1

    def mark_safe(self, cell): #DONE
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)



class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            if sentence.count > 0:
                sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            if sentence.count > 0:
                sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made - done
            2) mark the cell as safe - done
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """

        #issue: seems like "ignores known mines when adding new sentence" and "ignores known safes" suggests that maybe inference isn't deep enough
        #worth noting empty set may be not adding cells
        self.moves_made.add(cell)
        self.safes.add(cell)
        #need to find neighboring cells and use tuple for cells
        def neighbors(cell):
            #issue with neighboring, cannot go above
            cell_nn = set()
            y = cell[0]
            x = cell[1]
            for i in range(y-1, y+2):
                for j in range (x-1, x+2):
                    #print(f"{i},{j} i == height")
                    if i in range(0,self.height) and j in range(0,self.width):
                        if not(j == x and i==y):
                            cell_nn.add((i,j))
            return cell_nn


        new_sentence = Sentence(neighbors(cell),count)
        self.knowledge.append(new_sentence)
        cont = True
        n_iter = 0
        while cont == True:
            #may need a n_iter to give a grace period, ensure at least 3 iterations.

            #issue: seems like for some, added = 0
            out = []
            # used = []
            to_remove = []
            #finds all safe or all mines, and adds for use
            for c in self.safes:
                self.mark_safe(c)
            for c in self.mines:
                self.mark_mine(c)
            for i in self.knowledge:
                for j in self.knowledge:
                    a = i.cells
                    b = j.cells
                    if a.issubset(b):
                        new_cells = b-a
                        new_count = j.count - i.count
                        out.append(Sentence(new_cells, new_count))
                        cont = True
                        to_remove.append(i)
                        to_remove.append(j)
                    if b.issubset(a):
                        new_cell = a-b
                        new_count = i.count - j.count
                        out.append(Sentence(new_cell, new_count))
                        cont = True
                        to_remove.append(i)
                        to_remove.append(j)
                    if i.count == 1 and j.count == 1:
                        intersect = a.intersection(b)
                        xor_sets = a.symmetric_difference(b)
                        for m in self.mines:
                            if m in intersect:
                                out.append(Sentence(intersect,1))
                                out.append(Sentence(xor_sets,0))
                                cont = True
                                to_remove.append(i)
                                to_remove.append(j)
                            elif m in xor_sets:
                                out.append(Sentence(intersect,0))
                                out.append(Sentence(xor_sets,2))
                                cont = True
                                to_remove.append(i)
                                to_remove.append(j)
                            else:
                               pass
            for c in out:
                self.knowledge.append(c)
            print(len(self.knowledge))

            if len(to_remove) == 0 and n_iter > 5:
                cont = False
                print("terminating")
                #need a better termination conclusion, and add to cells
                #worth noting that it's a bit of an early terminator, what if none are passed? Need a grace period, that is, n_iter
                #But, how to determine a terminator?
            #assert len(self.knowledge) != 0 or n_iter <= 2
            #assert len(self.safes) != 0 or n_iter <= 2
            #assert len(self.mines) != 0 or n_iter <= 2
            n_iter += 1
            #issue: seems like self.mines is zeroed out and something or someone is removing from it
            for c in self.knowledge:
                if c.count == 0:
                    for i in c.cells:
                        self.safes.add(i)
                        to_remove.append(c)
                if c.count == len(c.cells):
                    for i in c.cells:
                        self.mines.add(i)
                        to_remove.append(c)

            for c in self.safes:
                self.mark_safe(c)
            for c in self.mines:
                self.mark_mine(c)

            for e in self.knowledge:
                if len(e.cells) == 0:
                    to_remove.append(e)
                    #apparently, does not remove an empty set
                if e.cells == set():
                    to_remove.append(e)

            for c in to_remove:
                if c in self.knowledge:
                    self.knowledge.remove(c)

            for point in self.knowledge:
                assert len(point.cells) != 0
                assert len(point.cells) > 0

            #cleans up all to remove


            #find a termination condition:
            #if len(to_remove) == 0, meaning that no more useful statements have been found, remove
            print(len(self.safes))
            print(len(self.mines))
        new_sentence = Sentence(neighbors(cell),count)
        self.knowledge.append(new_sentence)
        print(new_sentence)





    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        if len(self.moves_made) == len(self.safes):
            return None
        for c in self.moves_made:
            assert type(c) == tuple
            if Minesweeper().nearby_mines(c) == 0:
                def neighbors(cell):
                    #issue with neighboring, cannot go above
                    cell_nn = set()
                    y = cell[0]
                    x = cell[1]
                    for i in range(y-1, y+2):
                        for j in range (x-1, x+2):
                            #print(f"{i},{j} i == height")
                            if i in range(0,8) and j in range(0,8):
                                if not(j == x and i==y):
                                    cell_nn.add((i,j))
                    return cell_nn
                for i in neighbors(c):
                    self.safes.add(i)
                options = []
                for i in neighbors(c):
                    if i not in self.moves_made and i not in self.mines:
                        options.append(i)
                if options:
                    print('choosing near zero')
                    return random.choice(options)
        for c in self.safes:
            if c not in self.moves_made:
                return c



    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        #need to be set of all cells, so generate union
        total = set()
        for i in range(self.height):
            for j in range(self.width):
                total.add((i,j))
        do_not_access = self.mines.union(self.moves_made)
        possible = total - do_not_access
        return random.choice(list(possible))

if __name__ == "__main__":
    minesweeper = Minesweeper()
    minesweeper.print()
    ai = MinesweeperAI()
    for c in range(0,8):
        print("testing 5, " + str(c))
        if minesweeper.is_mine(cell = (5,c)) == False:
            ai.add_knowledge((5,c),minesweeper.nearby_mines((5,c)))
            print("finished")
            print(len(ai.knowledge))
            for i in ai.safes:
                print(f"{i}, {minesweeper.is_mine(i)}")
    print("search done")
    print(ai.make_safe_move())
    minesweeper.print()
    #find count


