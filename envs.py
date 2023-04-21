import numpy as np
import collections

mult_sign = "*"
sum_sign = "+"

class ProbGen:
    """
    This class samples arithmetic problems and generates their
    solutions as a series of steps following PEMDAS. The breakdown of
    each operations are as follows:
    
    Reorder:
        multiplications + ones + tens + 100s + ...

    Separate Mults:
        multiplications are progressively separated into sums

    Separate Into Respective Places:
        separate ones from 10s and 100s and ...
        separate 10s from 100s and 1000s and ...
        ...

    Sum ones:
        when ones exceed 10, separate into 10 and ones digit
        end by summing final ones to 10s
    Sum tens:
        when tens exceed 100, separate into 10s and 100s
        end by summing final tens to 100s
    Sum ...

    Incorrect Examples that I'm too lazy to correct:

        Sums:
            10+1+100=1+10+100=11+100=111
            22+10=32
            22+10+1=1+22+10=1+2+20+10=3+20+10=23+10=32
            13+25=3+10+25=3+5+10+20=8+10+20=8+30=38
            29+34=9+20+34=9+4+20+30=13+20+30=33+30=63
            27+75=7+20+75=5+7+20+70=12+20+70=32+70=102
            27+75+4=4+27+75=7+4+20+75=5+7+4+20+70=5+11+20+70=1+5+10+20+70=
                6+10+20+70=16+20+70=36+70=106
        
        Multiplication:
            6*6=5*6+6=4*6+6+6=3*6+6+6+6=2*6+6+6+6+6=6+6+6+6+6+6=
                6+6+6+6+12=2+6+6+6+6+10=2+6+6+12+10=2+2+12+10+10=
                2+2+2+10+10+10=2+4+10+10+10=6+10+10+10=6+10+10+10=
                16+10+10=26+10=36
            2*3=3+3=6
            5*3=3*5=2*5+5=5+5+5=5+10=15

        Both:
            13+6*6=6*6+13=5*6+6+13=
    """

    def __init__(self, max_num=10,
                       max_ents=3,
                       p_mult=0,
                       p_paren=0,
                       space_mults=True,
                       *args, **kwargs):
        """
        Args:
            max_num: int
                the maximum number available for sampling the initial
                problem
            max_ents: int
                maximum entities for the starting problem. If using
                parentheticals, a parenthetical counts as one entity,
                parentheticals are recursively samples with max_ents-1
                max entities.
            p_mult: float [0,1]
                the probability of sampling a multiplication sign for
                the starting problem
            p_paren: float [0,1]
                the probability of sampling a parenthetical.
                Parentheticals are sampled the same way as the initial
                problem but with max entities equal to max_ents-1
            space_mults: bool
                if true, will not allow more than two numbers to be
                multiplied together
        """
        self.max_num = max_num
        self.max_ents = max_ents
        self.p_mult = p_mult
        self.p_paren = p_paren
        self.space_mults = space_mults
        assert p_paren==0, "Parentheses are not yet implemented"

    @staticmethod
    def sample_prob(max_num, max_ents=2, p_mult=0, space_mults=True,
                                                   p_paren=0):
        """
        Args:
            max_num: int
                the maximum number available for sampling the initial
                problem
            max_ents: int
                maximum entities for the starting problem. If using
                parentheticals, a parenthetical counts as one entity,
                parentheticals are recursively samples with max_ents-1
                max entities.
            p_mult: float [0,1]
                the probability of sampling a multiplication sign for
                the starting problem
            space_mults: bool
                if true, will not allow more than two numbers to be
                multiplied together
            p_paren: float [0,1]
                the probability of sampling a parenthetical.
                Parentheticals are sampled the same way as the initial
                problem but with max entities equal to max_ents-1
        Returns:
            prob: str
                a string of a sampled arithmetic problem
        """
        prob = []
        op = sum_sign
        for i in range(max_ents):
            # 50% prob of not including more terms
            if i > 1 and np.random.random() > .5: continue
            if i > 0:
                if space_mults and op==mult_sign: op = sum_sign
                elif np.random.random()>p_mult:
                    op = sum_sign
                else: op = mult_sign
                prob.append(op)
            ent = str(np.random.randint(0,max_num+1))
            prob.append(ent)
            # Parentheticals
            if max_ents>2 and np.random.random()<p_paren:
                raise NotImplemented
                prob.append(ProbelmGen.sample_prob(
                    max_num,
                    max_ents=max_ents-1,
                    p_mult=p_mult,
                    p_paren=p_paren
                ))
        return "".join(prob)

    @staticmethod
    def entity_sort_key(entity):
        """
        Args:
            entity: str
        Returns:
            rank: int
                the sort rank of the entity
        """
        if "(" == entity[0]: return 0
        elif "*" in entity: return 1
        return len(entity) + 1

    @staticmethod
    def find_soln(prob, max_statements=np.inf):
        """
        Finds the algorithmic solution to the initial problem string.

        Args:
            prob: str
                a problem string as returned by `sample_prob`. Must
                only contain digits and + or * symbols.
            max_statements: int
                the maximum number of math statements
        Returns:
            soln: str
                the solution as dictated by the following algorithm
                starting from left to right, each step has its own
                statement. Does not include the initial problem but
                does include an initial = sign.
                  Step 1: 
                      - move parentheticals, multiplications, and
                          ones, tens, hundreds digits to left side of
                          statement in that order
                  Step 2 start first parenthetical and ones digits first:
                      - if only 1 selected digit, move to step 3
                      - else separate selected digits from next
                        magnitude (i.e. if ones is selected, separate
                        ones from tens), prepending selected digits
                        to left side of statement
                  Step 3:
                      - sum together ones digits starting with
                        rightmost ones digits.
                      - if there are remaining ones digits and a tens
                        digit forms, go back to step 1
                  Step 3:
                      - sum together 10s digits starting with
                        rightmost 10s digit
                      - if there are remaining 10s digits and a
                        hundreds digit forms, got back to step 1
        """
        statements = [prob]
        statement_set = set(statements)
        ents = statements[-1].split(sum_sign)
        ent_dict = ProbGen.get_ent_dict(ents)
        statement = ProbGen.make_statement(ent_dict)
        if statement not in statement_set:
            statement_set.add(statement)
            statements.append(statement)

        # Decompose Multiplication Terms
        loop = 0
        while len(ent_dict["mults"])>0:
            mult = ent_dict["mults"].popleft().split(mult_sign)
            ints = [ int(x) for x in mult ]
            # If we're down to a 1*x or 2*x we want to get rid of it
            if ints[0] == 0:
                ent_dict[1].appendleft("0")
            elif ints[0] <= 2:
                if len(ints)>2: # case of 1*x*y*...
                    mult = mult_sign.join(mult[1:])
                    for i in range(ints[0]):
                        ent_dict["mults"].appendleft(mult)
                else: # case of 1*x
                    for i in range(ints[0]):
                        ent_dict[len(mult[-1])].appendleft(mult[-1])
            else:
                ints[0] -= 1
                if len(ints)>2:
                    mult = mult_sign.join(mult[1:])
                    ent_dict["mults"].appendleft(mult)
                else:
                    ent_dict[len(mult[-1])].appendleft(mult[-1])
                mult = mult_sign.join([str(x) for x in ints])
                ent_dict["mults"].appendleft(mult)
            statement = ProbGen.make_statement(ent_dict)
            if statement not in statement_set:
                statement_set.add(statement)
                statements.append(statement)
            loop += 1
        if sum_sign not in statement and mult_sign not in statement:
            return "=".join(statements)

        # Progressively decompose into ones, tens, 100s, ...
        max_mag = len(str(eval_prob(statement)))
        for mag in range(1, max_mag):
            for j in range(mag+1,max_mag+1):
                if j in ent_dict:
                    for k,ent in enumerate(ent_dict[j]):
                        if ent[-mag] != "0":
                            # Remove Fragment and add to
                            # appropriate list
                            frag, ent = ProbGen.frag_ent(ent, mag)
                            ent_dict[mag].appendleft(frag)
                            ent_dict[j][k] = ent
                            statement = ProbGen.make_statement(ent_dict)
                            if statement not in statement_set:
                                statement_set.add(statement)
                                statements.append(statement)

        # Progressively sum each magnitude
        for mag in range(1,max_mag+1):
            if mag in ent_dict:
                while len(ent_dict[mag]) > 1:
                    ent1 = ent_dict[mag].popleft()
                    ent2 = ent_dict[mag].popleft()
                    ent = str(int(ent1)+int(ent2))
                    ent_dict[len(ent)].appendleft(ent)
                    # Record raw sum
                    statement = ProbGen.make_statement(ent_dict)
                    if statement not in statement_set:
                        statement_set.add(statement)
                        statements.append(statement)

                    # Decompose if remaining values of mag magnitude
                    if len(ent) != mag and len(ent_dict[mag])>0:
                        ent_dict[len(ent)].popleft()
                        frag, ent = ProbGen.frag_ent(ent, mag)
                        ent_dict[mag].appendleft(frag)
                        ent_dict[mag+1].appendleft(ent)
                    statement = ProbGen.make_statement(ent_dict)
                    if statement not in statement_set:
                        statement_set.add(statement)
                        statements.append(statement)
                if len(ent_dict[mag])==1:
                    ent1 = ent_dict[mag].pop()
                    for j in range(mag+1, max_mag+1):
                        if j in ent_dict and len(ent_dict[j])>0:
                            ent2 = ent_dict[j].popleft()
                            ent = str(int(ent1)+int(ent2))
                            ent_dict[j].appendleft(ent)
                            statement = ProbGen.make_statement(ent_dict)
                            if statement not in statement_set:
                                statement_set.add(statement)
                                statements.append(statement)
        return "=" + "=".join(statements[1:])

    @staticmethod
    def frag_ent(ent, mag):
        """
        Extracts the value at the argued magnitude. i.e. if the `ent` is
        '123', and the argued `mag` is 1, the returned values will be
        '120' and '3'. if `mag` was 1, then the returned would be '103' and
        '20'.
    
        Args:
            ent: str
                the entity to be fragmented
            mag: int
                the 10's place to extract from. 1 means ones place. 2
                means 10s and so on
        Returns:
            frag: str
                the extracted magnitude
            ent: str
                the original entity without the value at the argued mag
        """
        if mag > len(ent): return "0", ent # this shouldn't happen
        frag = ent[-mag] + "0"*(mag-1)
        if mag==1: ent = ent[:-1]+"0"
        else: ent = ent[:-mag]+"0"+ent[-mag+1:]
        return frag, ent

    @staticmethod
    def make_statement(ent_dict, max_mag=4):
        """
        Args:
            sort_dict: dict of list of str
                a dict with keys of the character length of each integer
                with values of a list of str that fall into that integer
                length.

                "mult": list of str
                    all entities with multiplication signs in them
                1: list of str
                    all entities with a length of 1 (ones integers)
                2: list of str
                    all entities with a length of 2 (tens integers)
            max_mag: int or None (optional)
                the maximum magnitude of the problem
        Return:
            statement: str
                a mathematical statement
        """
        ents = [x for x in ent_dict["mults"]]
        if max_mag is None:
            keys = set(ent_dict.keys())
            del keys["mults"]
            max_mag = max(keys)
        keys = np.arange(max_mag)
        for k in keys:
            if k in ent_dict and k != "mults": ents += ent_dict[k]
        return sum_sign.join(ents)

    @staticmethod
    def get_ent_dict(ents):
        """
        Sorts the entities into a dict. Also sorts the multiplication
        entities into smallest*largest

        Args:
            ents: list of str
                list of entities

        Returns:
            sort_dict: dict of list of str
                a dict with keys of the character length of each integer
                with values of a list of str that fall into that integer
                length.

                "mult": list of str
                    all entities with multiplication signs in them
                1: list of str
                    all entities with a length of 1 (ones integers)
                2: list of str
                    all entities with a length of 2 (tens integers)
        """
        sort_dict = collections.defaultdict(collections.deque)
        for ent in ents:
            # TODO: Parentheticals are not yet implemented
            if mult_sign in ent:
                ent = ProbGen.sort_mult(ent)
                sort_dict["mults"].append(ent)
            else:
                sort_dict[len(ent)].append(ent)
        return sort_dict

    @staticmethod
    def sort_mult(ent):
        """
        Sorts a multiplication entity from smallest term to largest.

            5*2*3 -> 2*3*5

        Args:
            ent: str
        Returns:
            ent: str
        """
        integers = sorted(ent.split(mult_sign), key=lambda x: int(x))
        return mult_sign.join(integers)

def eval_prob(prob):
    """
    prob: str
        contains digits, + and *
    Returns:
        eval: int
            evaluated expression
    """
    ents = []
    for ent in prob.split(sum_sign):
        if mult_sign in ent:
            ent = np.prod([int(x) for x in ent.split(mult_sign)])
        ents.append(int(ent))
    return np.sum(ents)


if __name__=="__main__":
    max_len = 0
    min_len = 100
    for i in range(100):
        prob = ProbGen.sample_prob(
            max_num=20,
            max_ents=3,
            p_mult=0.5,
            space_mults=True
        )
        soln = ProbGen.find_soln(prob)
        try:
            if len(soln) < min_len:
                min_len = len(soln)
                min_soln = soln
            elif len(soln) > max_len:
                max_len = len(soln)
                max_soln = soln
            print(soln)
            print()
            gtruth = eval_prob(prob)
            splt2 = soln.split("=")[-1]
            assert gtruth == int(splt2)
        except:
            print("gtr:", gtruth)
            print("try:", splt2)
            print(soln)
            assert False
    print("Min:",min_len)
    print(min_soln)
    #print("Max:",max_len)
    #print(max_soln)

    #prob = ProbGen.sample_prob(
    #    max_num=20,
    #    max_ents=3,
    #    p_mult=0
    #)
    #soln = ProbGen.find_soln(prob)
    #print("Soln:", soln)
    #splt = [int(x) for x in prob.split("+")]
    #splt2 = soln.split("=")[-1]
    #print(splt)
    #print(splt2)
    #assert np.sum(splt) == int(splt2)
    #print()




