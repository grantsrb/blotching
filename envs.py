import numpy as np
import collections

mult_sign = "*"
sum_sign = "+"
"""
REARRANGE:
The rearrange operation reorders the entities within the equation so
that the magnitudes are in progressive order. It does not change the
relative ordering of values within a magnitude.

DECOMPOSE:
The decomposition operation separates the value of a particular magnitude
from a greater magnitude. So, for instance, 101=1+100 would be a
decompose3->1, and 110=10+100 would be decompose3->2

SUM
The sum operation applies only to values within the same magnitude. For
instance, 4+5=9 is a sum1->1, whereas 5+6=11 is a sum1->2, and 20+90=110
is a sum2->3. Summing values that are of different magnitudes is called
a COMBINE operation.

COMBINE
Similar to sum, but combine refers to summing values of different
magnitudes. For instance 10+100=110 is a combine2->3 operation, whereas
1+20=21 is a combine1->2 operation.
"""
REARRANGE = "rearrange"
DECOMPOSE = "decompose"
# used when two numbers of the same magnitude are summed
SUM = "sum"
COMBINE = "combine"
SOLN_TYPES = [ REARRANGE, DECOMPOSE, SUM, COMBINE ]

class MathEnv:
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
                       max_mult_num=10,
                       zipf_order=0,
                       p_ent=0.5,
                       *args, **kwargs):
        """
        Args:
            max_num: int
                the maximum number available for sampling the initial
                problem (inclusive)
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
            zipf_order: float
                the exponent of a zipfian distribution by which to
                sample each entity.
            p_ent: float [0,1]
                the probability of samping each entity beyond the first
                two. There will always be at least two entities, but
                beyond that, each entity has a p_ent chance of being
                sampled at all. In the case that each entity fails to
                be sampled, the resulting problem has fewer entities.
                A value of 1 means each entity is guaranteed to be
                sampled, a value of 0 is equivalent to setting the
                max_ents to 2.
            max_mult_num: int
                the maximum value to be multiplied with. This applies
                to only one of the two numbers involved in the
                multiplication. For example, in x*y, if x is sampled
                and happens to be larger than max_mult_num, then y
                will be sampled from the range [0, max_mul_num]. If
                x happens to be sampled as less than or equal to
                max_mult_num, then y will be sampled from [0,max_num]
        """
        self.max_num = max_num
        self.max_ents = max_ents
        self.p_mult = p_mult
        self.max_mult_num = max_mult_num
        if max_mult_num is None: self.max_mult_num = self.max_num
        self.p_paren = p_paren
        assert p_paren==0, "Parentheses are not yet implemented"
        self.space_mults = space_mults
        self.zipf_order = zipf_order
        if p_ent is None: self.p_ent = (self.max_num-1)/self.max_num
        else: self.p_ent = p_ent
        self.max_prob = self.get_max_prob()
        print("Max Prob:", self.max_prob)
        self.prob_len = len(self.max_prob)
        prob,self.max_soln = self.get_max_soln()
        print("Max Soln:", prob + "=" + self.max_soln)
        print(
            "WARNING: It's potentially possible that this isn't the"+\
            " maximum length solution.."
        )
        self.max_soln_len = len(self.max_soln)

    def get_max_prob(self):
        """
        Creates the maximum possible length problem given the member
        variables. This does not necessarily result in the max_soln

        Returns:
            max_prob: str
                the maximum length problem in terms of characters
        """
        max_prob = sum_sign.join(
            [str(self.max_num) for _ in range(self.max_ents)]
        )
        return max_prob

    def get_max_soln(self):
        """
        Creates the maximum possible length solution given the member
        variables. This does not necessarily use the max_prob

        Returns:
            max_soln: str
                the maximum length problem in terms of characters
        """
        ent = int("".join(["9" for _ in range(len(str(self.max_num))-1)]))
        prob = sum_sign.join(
            [str(ent) for _ in range(self.max_ents)]
        )
        soln = MathEnv.find_soln(prob)
        if self.p_mult>0:
            mult_prob = []
            mmn = self.max_mult_num
            if mmn%10==0: mmn = mmn-1
            mult_ent = int("".join([
              str(mmn) for _ in range(max(len(str(self.max_mult_num))-1,1))
            ]))
            mult_prob.append(mult_ent)
            i = 1
            while i < self.max_ents:
                mult_prob.append(mult_sign)
                mult_prob.append(mult_ent)
                i += 1
                if i >= self.max_ents: break
                if mult_prob[-2]==mult_sign and self.space_mults:
                    mult_prob.append(sum_sign)
                    mult_prob.append(ent)
                    i+=1
            mult_prob = "".join([str(p) for p in mult_prob])
            mult_soln = MathEnv.find_soln(mult_prob)
            if len(mult_soln)>len(soln):
                soln = mult_soln
                prob = mult_prob
        return prob,soln

    def sample(self):
        """
        Samples a problem using the parameters specific to self
        """
        return MathEnv.sample_prob(
            max_num=self.max_num,
            max_ents=self.max_ents,
            p_mult=self.p_mult,
            space_mults=self.space_mults,
            p_paren=self.p_paren,
            zipf_order=self.zipf_order,
            p_ent=self.p_ent,
            max_mult_num=self.max_mult_num,
        )

    @staticmethod
    def sample_prob(max_num, max_ents=2, p_mult=0, space_mults=True,
                                                   p_paren=0,
                                                   zipf_order=0,
                                                   max_mult_num=None,
                                                   p_ent=0.5):
        """
        Args:
            max_num: int
                the maximum number available for sampling the initial
                problem (inclusive)
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
            zipf_order: float
                the exponent of a zipfian distribution by which to sample
                each entity from.
            p_ent: float [0,1]
                the probability of samping each entity beyond the first
                two. There will always be at least two entities, but
                beyond that, each entity has a p_ent chance of being
                sampled at all. In the case that each entity fails to
                be sampled, the resulting problem has fewer entities.
                A value of 1 means each entity is guaranteed to be
                sampled, a value of 0 is equivalent to setting the
                max_ents to 2.
            max_mult_num: int
                the maximum value to be multiplied with. This applies
                to only one of the two numbers involved in the
                multiplication. For example, in x*y, if x is sampled
                and happens to be larger than max_mult_num, then y
                will be sampled from the range [0, max_mul_num]. If
                x happens to be sampled as less than or equal to
                max_mult_num, then y will be sampled from [0,max_num]
        Returns:
            prob: str
                a string of a sampled arithmetic problem
        """
        prob = []
        op = sum_sign
        high_num = max_num
        if max_mult_num is None: max_mult_num = max_num
        break_i = max_ents
        for i in range(max_ents):
            # 50% prob of not including more terms
            if i > 1 and np.random.random() < p_ent: continue
            if i > 0:
                high_num = max_num
                if space_mults and op==mult_sign: op = sum_sign
                elif np.random.random()>p_mult:
                    op = sum_sign
                else:
                    op = mult_sign
                    high_num = max_mult_num
                prob.append(op)
            if zipf_order>0:
                ent = int(zipfian(
                    low=1,high=high_num+1,order=zipf_order
                ))
                if op == mult_sign:
                    prob[-2] = ent
                    ent = int(zipfian(
                        low=1,high=high_num+1,order=zipf_order
                    ))
            else:
                ent = np.random.randint(0,high_num+1)
                if op == mult_sign:
                    prob[-2] = ent
                    ent = np.random.randint(0,high_num+1)
            prob.append(ent)
            # 50% chance to flip multiplication entities to try to
            # reduce bias of sampling.
            if op==mult_sign and np.random.random()>0.5:
                prob[-3],prob[-1] = prob[-1], prob[-3]
            # Parentheticals
            if max_ents>2 and np.random.random()<p_paren:
                raise NotImplemented
                prob.append(ProbelmGen.sample_prob(
                    max_num,
                    max_ents=max_ents-1,
                    p_mult=p_mult,
                    p_paren=p_paren
                ))
        return "".join([str(p) for p in prob])

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
    def find_soln(prob, max_statements=np.inf, ret_labels=False):
        """
        Finds the algorithmic solution to the initial problem string.

        Args:
            prob: str
                a problem string as returned by `sample_prob`. Must
                only contain digits and + or * symbols.
            max_statements: int
                the maximum number of math statements
            ret_labels: bool
                if true, will return a list that indicates the type
                of operation that was performed to arrive at the
                corresponding step. Thus the length of the list is
                equal to the number of equals signs + 1.
        Returns:
            labels: list of str
                a list of operation labels, only returned if ret_labels
                is true.
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
        labels = []
        statement_set = set(statements)
        ents = statements[-1].split(sum_sign)
        ent_dict = MathEnv.get_ent_dict(ents)
        statement = MathEnv.make_statement(ent_dict)
        if statement not in statement_set:
            statement_set.add(statement)
            statements.append(statement)
            labels.append(REARRANGE)

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
            statement = MathEnv.make_statement(ent_dict)
            if statement not in statement_set:
                statement_set.add(statement)
                statements.append(statement)
                labels.append(DECOMPOSE+"mult")
            loop += 1
        if sum_sign not in statement and mult_sign not in statement:
            return "=".join(statements)

        # Progressively decompose into ones, tens, 100s, ...
        # but only when there are multiple of a particular magnitude.
        max_mag = len(str(eval_prob(statement)))
        for mag in range(1, max_mag):
            # First Check if decomposition is needed
            n_frags = len(ent_dict[mag])
            for j in range(mag+1,max_mag+1):
                if j in ent_dict:
                    for k,ent in enumerate(ent_dict[j]):
                        if ent[-mag] != "0":
                            n_frags += 1
            if n_frags <= 1: continue

            for j in range(mag+1,max_mag+1):
                if j in ent_dict:
                    for k,ent in enumerate(ent_dict[j]):
                        if ent[-mag] != "0":
                            # Remove Fragment and add to
                            # appropriate list
                            frag, ent = MathEnv.frag_ent(ent, mag)
                            ent_dict[mag].appendleft(frag)
                            ent_dict[j][k] = ent
                            statement = MathEnv.make_statement(ent_dict)
                            if statement not in statement_set:
                                statement_set.add(statement)
                                statements.append(statement)
                                labels.append(
                                    DECOMPOSE+"{}->{}".format(j,mag)
                                )
        #print("Through decomp")
        #print("=".join(statements))
        #print("Beginning Summation")

        # Progressively sum each magnitude
        for mag in range(1,max_mag+1):
            if mag in ent_dict:
                #print("Beginning mag", mag)
                while len(ent_dict[mag]) > 1:
                    ent1 = ent_dict[mag].popleft()
                    ent2 = ent_dict[mag].popleft()
                    ent = str(int(ent1)+int(ent2))
                    ent_dict[len(ent)].appendleft(ent)
                    # Record raw sum
                    statement = MathEnv.make_statement(ent_dict)
                    if statement not in statement_set:
                        statement_set.add(statement)
                        statements.append(statement)
                        labels.append(SUM+"{}->{}".format(mag,len(ent)))

                    # Decompose if remaining values of mag magnitude
                    if len(ent) != mag and len(ent_dict[mag])>0 and\
                                                    ent[-mag] != "0":
                        #print("Inside if")
                        #print("ent:", ent)
                        #print("mag:", mag)
                        #print("entmag-1:", ent[mag-1])
                        #print("len ent dict:", len(ent_dict[mag]))
                        ent_dict[len(ent)].popleft()
                        frag, ent = MathEnv.frag_ent(ent, mag)
                        ent_dict[mag].appendleft(frag)
                        ent_dict[mag+1].appendleft(ent)

                        statement = MathEnv.make_statement(ent_dict)
                        if statement not in statement_set:
                            statement_set.add(statement)
                            statements.append(statement)
                            labels.append(
                                DECOMPOSE+"{}->{}".format(len(ent),mag)
                            )
                if len(ent_dict[mag])==1:
                    #print("inside final loop", mag, statements[-1])
                    ent1 = ent_dict[mag].pop()
                    for j in range(mag+1, max_mag+1):
                        if j in ent_dict and len(ent_dict[j])>0:
                            ent2 = ent_dict[j].popleft()
                            ent = str(int(ent1)+int(ent2))
                            ent_dict[j].appendleft(ent)
                            statement = MathEnv.make_statement(ent_dict)
                            if statement not in statement_set:
                                #print("adding statement", statement)
                                statement_set.add(statement)
                                statements.append(statement)
                                labels.append(
                                    COMBINE+"{}->{}".format(mag,j)
                                )
                            break
        soln = "=".join(statements[1:])
        if ret_labels: return soln, labels
        return soln

    @staticmethod
    def frag_ent(ent, mag):
        """
        Extracts the value at the argued magnitude. i.e. if the `ent` is
        '123', and the argued `mag` is 1, the returned values will be
        '120' and '3'. if `mag` was 2, then the returned would be '103'
        and '20'.
    
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
    def make_statement(ent_dict, max_mag=None):
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
            keys.remove("mults")
            if len(keys)>0:
                max_mag = max(keys)
        if max_mag is not None:
            keys = np.arange(max_mag+1)
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
                ent = MathEnv.sort_mult(ent)
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

    @staticmethod
    def recursive_probs(prob="", n_ents=3, max_num=100, all_probs=[]):
        """
        Returns a list of problem strings ranging from 1+1 to
        max_num+max_num+... for n_ents.

        Args:
            prob: str
                the problem string so far.
            n_ents: int
                the number of entities remaining to handle.
            max_num: int (inclusive)
                the inclusive maximum value to include in the problem
                generation.
            all_probs: list
                the list to be populated with new problem strings
        Returns:
            all_probs: list of str
                all problem strings from 1+1 to max_num+max_num+...
        """
        if n_ents == 0:
            return all_probs
        for i in range(max_num+1):
            if len(prob)>0:
                new_prob = prob + sum_sign + str(i)
                all_probs.append(new_prob)
            else: new_prob = str(i)
            all_probs = MathEnv.recursive_probs(
                new_prob, n_ents-1, max_num, all_probs=all_probs
            )
        return all_probs

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

def zipfian(low=1, high=9, order=1, size=None):
    """
    Draws a single integer from low (inclusive) to high (inclusive) in
    which the probability is proportional to 1/k^order.

    Args:
        low: int (inclusive)
            the lowest possible value
        high: int (exclusive)
            one less than the highest possible value
        order: float
            the order of the exponent to weight the probability density
            for each possible value.
    Returns:
        sample: int
            returns a sample drawn from the zipfian distribution.
    """
    if low == high-1: return low
    assert low < high-1 and low > 0
    nums = np.arange(low, high)
    probs = 1./(nums**order)
    probs = probs/probs.sum()
    samp = np.random.choice(nums, p=probs, size=size)
    return samp


if __name__=="__main__":
    #probs = MathEnv.recursive_probs(
    #    "", 4, 5
    #)
    #for prob in probs:
    #    print(prob)
    #    soln = MathEnv.find_soln(prob)
    #    print("Soln:", soln)
    math_env = MathEnv(
            max_num=100,
            max_ents=3,
            p_mult=0.5,
            p_paren=0,
            space_mults=True,
            max_mult_num=6,
            zipf_order=0,
            p_ent=0.5,
    )
    #max_len = 0
    #min_len = 100
    #prob_hist = collections.defaultdict(lambda: 0)
    #for i in range(1000):
    #    prob = MathEnv.sample_prob(
    #        max_num=100,
    #        max_ents=3,
    #        p_mult=0.25,
    #        space_mults=True,
    #        max_mult_num=5
    #    )
    #    if "*" in prob:
    #        splt = prob.split(sum_sign)
    #        for s in splt:
    #            prob_hist[s] += 1
    #    soln = MathEnv.find_soln(prob)
    #    if "=00" in soln:
    #        print()
    #        print("Found issue:")
    #        print("prob:", prob)
    #        print("Soln:", soln)
    #    try:
    #        if len(soln) < min_len:
    #            min_len = len(soln)
    #            min_soln = soln
    #            min_prob = prob
    #        elif len(soln) > max_len:
    #            max_len = len(soln)
    #            max_soln = soln
    #            max_prob = prob
    #        #print(soln)
    #        #print()
    #        gtruth = eval_prob(prob)
    #        splt = soln.split("=")
    #        for s in splt:
    #            assert gtruth == eval_prob(s)
    #    except:
    #        print("gtr:", gtruth)
    #        print("try:", splt2)
    #        print(soln)
    #        assert False
    #print("Min:",min_len)
    #print("prob:", min_prob)
    #print("soln:", min_soln)
    #print()
    #print("Max:",max_len)
    #print("prob:", max_prob)
    #print("soln:", max_soln)
    ##print()
    ##print("Hist:")
    ##keys = sorted(list(prob_hist.keys()))
    ##for k in keys:
    ##    print("\t{}: {}".format(k,prob_hist[k]))

    ###prob = MathEnv.sample_prob(
    ###    max_num=20,
    ###    max_ents=3,
    ###    p_mult=0.5,
    ###    max_mult_num=10,
    ###)
    ###soln = MathEnv.find_soln(prob)
    ###print("Soln:", soln)
    ###splt = [int(x) for x in prob.split("+")]
    ###splt2 = soln.split("=")[-1]
    ###print(splt)
    ###print(splt2)
    ###assert np.sum(splt) == int(splt2)
    ###print()




