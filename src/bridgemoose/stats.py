import math
from collections import Counter
from collections import defaultdict
import time

from . import auction
from . import dds
from . import direction
from . import random
from . import scoring

class Statistic:
    def __init__(self):
        self.count = 0
        self.sum_weight = 0.0
        self.sum_weight_sq = 0.0
        self.mean = 0.0
        self.sum_squares = 0.0

    def add_data_point(self, x, weight=1.0):
        self.count += 1
        self.sum_weight += weight
        self.sum_weight_sq += weight * weight

        mean_old = self.mean
        delta = x - mean_old

        self.mean += (weight / self.sum_weight) * delta
        self.sum_squares += weight * delta * (x - self.mean)

    def sample_variance(self):
        if self.sum_squares == 0:
            return 0
        else:
            return self.sum_squares / (self.sum_weight - self.sum_weight_sq / self.sum_weight)

    def std_error(self):
        if self.sum_weight <= 0:
            return 0.0
        return math.sqrt(self.sample_variance() * self.sum_weight_sq) / self.sum_weight

    def __str__(self):
        return "%.2f +/- %.2f" % (self.mean, self.std_error())
        
def dd_compare_strategies(deal_generator, strategy1, strategy2,
    score_type="imps", vulnerability=None, bucketer=None):
    """\
Returns a triple; a Statistic and two collections.Counter objects.
The Statistic returns the score from the NS point of view of strategy1
vs. strategy2.  The Counters are essentially dicts mapping ("<contract-declarer>", score) pairs to counts.

A "strategy" can be either a contract-declarer string ("3NT-W") or a
function which takes a Deal and returns a contract-declarer.

"Scoring" can be 'imps', 'matchpoints', 'TOTAL'.

"vulnerability" can be None, "", "NS", "EW", "BOTH", "NSEW"

If bucketer is not None, it is a function which takes a Deal and returns
a dict key, and then dd_compare_strategies returns a dict whose values
are triples as described above.
    """
    if vulnerability is None:
        vul = ""
    elif vulnerability in ("BOTH", "b"):
        vul = "NSEW"
    elif vulnerability in ("", "NS", "EW", "NSEW", "EWNS"):
        vul = vulnerability
    else:
        raise ValueError("Vulnerability should be '', 'NS', 'EW', or 'NSEW'")

    score_type = score_type.lower()
    if score_type in ('imps', 'imp'):
        score_func = scoring.scorediff_imps
    elif score_type in ('mps', 'mp', 'matchpoints'):
        score_func = scoring.scorediff_matchpoints
    elif score_type in ('total'):
        score_func = lambda x: x
    else:
        raise ValueError("Scoring should be one of 'imps','matchpoints','total'")

    sign = {"N":1, "S":1, "E":-1, "W":-1}

    out = defaultdict(lambda: (Statistic(), Counter(), Counter()))

    deals = []
    queries = []
    for deal in deal_generator:
        deals.append(deal)

        cd1 = strategy1(deal) if callable(strategy1) else strategy1
        cd2 = strategy2(deal) if callable(strategy2) else strategy2
        con1, dec1 = cd1.split("-")
        con2, dec2 = cd2.split("-")

        strain1 = con1[1]
        strain2 = con2[1]

        if (strain1, dec1) == (strain2, dec2):
            queries.append((deal, dec1, strain1))
        else:
            queries.append((deal, dec1, strain1))
            queries.append((deal, dec2, strain2))

    answers = []
    for i in range(0, len(queries), 200):
        answers.extend(dds.solve_many_deals(queries[i:i+200]))

    for deal in deals:
        cd1 = strategy1(deal) if callable(strategy1) else strategy1
        cd2 = strategy2(deal) if callable(strategy2) else strategy2
        con1, dec1 = cd1.split("-")
        con2, dec2 = cd2.split("-")

        strain1 = con1[1]
        strain2 = con2[1]

        if (strain1, dec1) == (strain2, dec2):
            tx1 = answers.pop(0)
            tx2 = tx1
        else:
            tx1 = answers.pop(0)
            tx2 = answers.pop(0)

        score1 = sign[dec1] * scoring.result_score(con1,
            tx1, dec1 in vul)
        score2 = sign[dec2] * scoring.result_score(con2,
            tx2, dec2 in vul)

        b = "" if bucketer is None else bucketer(deal)
        stat, count1, count2 = out[b]

        stat.add_data_point(score_func(score1 - score2))
        count1[(cd1, score1)] += 1
        count2[(cd2, score2)] += 1

    if bucketer is None:
        return out[""]
    else:
        return out

class ReportingTimer:
    def __init__(self, name=None):
        self.name = name
        self.count = 0
        self.last_start = None
        self.last_check = None
        self.total_time = 0

    @staticmethod
    def fmt_elapsed(amount):
        if amount < 1:
            return f"{amount:.3f}s"
        elif amount < 10:
            return f"{amount:.2f}s"
        elif amount < 100:
            return f"{amount:.1f}s"
        else:
            return f"{amount:.0f}s"

    def __enter__(self):
        self.count += 1
        self.last_start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.time() - self.last_start
        self.total_time += elapsed
        pstr = f"{self.name if self.name else 'Timer'}: "
        if self.count == 1:
            pstr += self.fmt_elapsed(elapsed)
        else:
            pstr += f"avg {self.fmt_elapsed(self.total_time/self.count)} over {self.count} calls"
        print(pstr)

        
class DDComparison:
    def __init__(self, contracts1, scores1, contracts2, scores2, score_type,
                 strat_names=None):
        self.contracts1 = contracts1
        self.scores1 = scores1
        self.contracts2 = contracts2
        self.scores2 = scores2
        self.score_type = score_type.lower()
        if strat_names is None:
            self.strat_names = ["Strat 1", "Strat 2"]
        else:
            self.strat_names = [f"Strat {i+1}" if x is None else x
                                for i, x in enumerate(strat_names)]

    def contract_counter(self, index):
        return Counter([self.contracts1, self.contracts2][index])

    def score_counter(self, index):
        return Counter([self.scores1, self.scores2][index])

    def advantage_1(self, score_type=None):
        if score_type is None:
            score_type = self.score_type
        else:
            score_type = score_type.lower()

        scorers = {
            "imps": scoring.scorediff_imps,
            "imp": scoring.scorediff_imps,
            "mps": scoring.scorediff_matchpoints,
            "mp": scoring.scorediff_matchpoints,
            "matchpoints": scoring.scorediff_matchpoints,
            "total": lambda x: x,
        }
        score_func = scorers[score_type]

        stat = Statistic()

        for s1, s2 in zip(self.scores1, self.scores2):
            stat.add_data_point(score_func(s1 - s2))

        return stat

    def display(self, method="full"):
        if method not in ["full","short"]:
            raise ValueError("method should be 'full' or 'short'")

        stat = self.advantage_1()

        scale = {"matchpoints":100, "imps":1, "total":1}[self.score_type]
        sum_means = {"matchpoints":1, "imps":0, "total":0}[self.score_type]
        print(f"Advantage for {self.strat_names[0]}: {scale*stat.mean:5.2f} +/- {scale*stat.std_error():5.2f}{'':10} {self.strat_names[1]}: {scale*(sum_means-stat.mean):5.2f}")

        if method == "short":
            return

        for index in range(2):
            cc = [(num, con) for con, num in self.contract_counter(index).items()]
            print()
            print(f"Contracts for {self.strat_names[index]}")
            for val, con in sorted(cc):
                print(f"{str(con):8} {val:8}")
            print()
            print(f"Scores for Strat {self.strat_names[index]}")
            sc = sorted(list(self.score_counter(index).items()), reverse=True)
            for score, num in sc:
                print(f"{score:8} {num:8}")


class DDAnalyzer:
    def __init__(self, count, west=None, north=None, east=None, south=None,
        accept=None, rng=None, score_type="imps", vulnerability=None):
        #
        self.count = count
        self.gen = random.RestrictedDealer(west, north, east, south, accept, rng)
        self.deals = []
        self.tricks = []
        self.add_deals(count)
        self.score_type = score_type
        self.vulnerability = vulnerability

    def add_deals(self, count):
        misses = 0
        hits = 0
        while hits < count:
            deal = self.gen.one_try()
            if deal is None:
                misses += 1
                if hits == 0 and misses > 100000:
                    raise ValueError("100,000 misses.  Perhaps there are no valid deals?")
            else:
                self.deals.append(deal)
                self.tricks.append({})
                hits += 1

    def test_strategy(self, strategy, show="NS"):
        dc_list = self._get_contracts(strategy)
        for deal, dc in zip(self.deals, dc_list):
            print(f"{' '.join(f'{h}={deal[h]:17}' for h in show)} {dc}")


    def _get_contracts(self, strategy, ignore_func=None):
        if callable(strategy):
            return [None if ignore_func and ignore_func(deal) else auction.DeclaredContract(strategy(deal)) for deal in self.deals]
        else:
            return [None if ignore_func and ignore_func(deal) else auction.DeclaredContract(strategy) for deal in self.deals]

    @staticmethod
    def _strategy_name(strategy):
        if callable(strategy):
            if strategy.__name__ == "<lambda>":
                return None
            else:
                return strategy.__name__
        else:
            return str(strategy)

    def compare(self, strategy1, strategy2, score_type=None,
        vulnerability=None, ignore_func=None):
        #
        contracts1 = self._get_contracts(strategy1, ignore_func)
        contracts2 = self._get_contracts(strategy2, ignore_func)

        self._dds_work(contracts1, contracts2)

        # first default value
        if vulnerability is None:
            vulnerability = self.vulnerability

        if vulnerability is None:
            vulnerability = ""
        elif vulnerability in ("BOTH", "b"):
            vulnerability = "NSEW"

        vul_set = [direction.Direction(x) for x in vulnerability]

        scores1 = self._score_up(contracts1, vul_set)
        scores2 = self._score_up(contracts2, vul_set)

        strat_names = [
            self._strategy_name(strategy1),
            self._strategy_name(strategy2),
        ]

        return DDComparison(contracts1, scores1, contracts2, scores2,
            self.score_type if score_type is None else score_type,
                            strat_names=strat_names)
    compare_strategies = compare

    def _score_up(self, contracts, vul_set):
        sign = {
            direction.Direction.NORTH: 1,
            direction.Direction.SOUTH: 1,
            direction.Direction.EAST: -1,
            direction.Direction.WEST: -1,
        }

        return [sign[con.declarer] * scoring.result_score(con,
            self.tricks[i][con.ds()], con.declarer in vul_set) for
            i, con in enumerate(contracts) if con is not None]

    def _dds_work(self, contracts1, contracts2):
        work = []
        work_index = []
        for i in range(len(self.deals)):
            if contracts1[i] is None or contracts2[i] is None:
                continue
            DS1 = contracts1[i].ds()
            DS2 = contracts2[i].ds()
            if not DS1 in self.tricks[i]:
                work.append((self.deals[i], DS1[0], DS1[1]))
                work_index.append((i, DS1))
            if DS1 != DS2 and not DS2 in self.tricks[i]:
                work.append((self.deals[i], DS2[0], DS2[1]))
                work_index.append((i, DS2))

        print(f"DEBUG: amount of work is {len(work)}")
        answers = dds.solve_many_deals(work)

        for answer, (deal_num, ds) in zip(answers, work_index):
            self.tricks[deal_num][ds] = answer
