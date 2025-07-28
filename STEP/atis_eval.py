import re
from enum import Enum
from typing import List, Dict, Union, Set, Tuple

from nltk import Tree, ImmutableTree

s = re.compile("([,() ])")

quotes = re.compile("(')")

import Levenshtein

def split_tokens(x):
    return [x for x in s.split(x.strip()) if x and x.strip(" ")]


def to_lisp(toks):
    r = []
    for t in toks:
        if t == "(":
            r = r[:-1] + ["("] + [r[-1]]
        elif t == ",":
            r.append(" ")
        else:
            r.append(t)
    return r


def get_tree(s):
    s = split_tokens(s)
    return Tree.fromstring(" ".join(to_lisp(s)))


def tree2funql(t):
    if isinstance(t, str):
        return t
    return t.label() + "(" + ",".join(tree2funql(c) for c in t) + ")"


# def preorder_with_brackets(t, ambiguous_arities):
#     children = list(t)
#     crs = [preorder_with_brackets(c, ambiguous_arities) if isinstance(c, Tree) else c for c in children]
#     if len(crs) == 0:
#         return t
#     if t.label() in ambiguous_arities:
#         return "( " + t.label() + " " + " ".join(crs) + " )"
#     else:
#         return t.label() + " " + " ".join(crs)


class QuoteHandling(Enum):
    NOOP = 0
    SEPARATE = 1
    DELETE = 2


def preorder_wo_brackets(t: Union[str, Tree], quote_handling: QuoteHandling) -> List[str]:
    if isinstance(t, str):
        if quote_handling == QuoteHandling.SEPARATE:
            return [x for x in quotes.split(t) if x]
        elif quote_handling == QuoteHandling.DELETE:
            return [t.replace("'", "")]
        else:
            return [t]
    children = list(t)
    r = [t.label()]
    for c in children:
        r.extend(preorder_wo_brackets(c, quote_handling))
    return r


def reconstruct_tree_without_brackets(s: List[str], arities: Dict[str, int]) -> Tree:
    """
    Reconstruct a tree from a linearized representation without brackets.
    :param s:
    :param arities:
    :param add_quotes:
    :param joiner:
    :return:
    """
    position = 0

    def read_tree():
        nonlocal position
        head = s[position]
        position += 1
        if head in arities:
            return Tree(head, children=[read_tree() for _ in range(arities[head])])
        else:
            return head  # assume arity 0

    t = read_tree()
    assert position == len(s)
    # ~ print(position, len(s))
    return t


def reconstruct_tree_with_partial_brackets(s: List[str], arities: Dict[str, int]) -> Tree:
    position = 0

    def read_tree():
        nonlocal position
        head = s[position]
        position += 1
        if head == "(":
            node_name = s[position]
            position += 1
            trees = []
            while s[position] != ")":
                trees.append(read_tree())
            position += 1
            return Tree(node_name, children=trees)

        elif head in arities:
            return Tree(head, children=[read_tree() for _ in range(arities[head])])
        else:
            return head

    t = read_tree()
    assert position == len(s)
    return t


def join_quotes(s: List[str]) -> List[str]:
    """
    Compresses a sequence of unknown tokens into a single element (a tuple).
    :param s:
    :param known:
    :return:
    """
    last_part = ""
    r = []
    in_quotes = False
    for i in range(len(s)):
        if in_quotes:
            last_part += " " + s[i]
            if s[i].endswith("'"):
                in_quotes = False
                r.append(last_part)
                last_part = ()
        else:

            if s[i].startswith("'") == s[i].endswith("'"):  # if no quotes or quotes on both sides
                r.append(s[i])
            elif s[i].startswith("'"):
                last_part = s[i]
                in_quotes = True

    if last_part:
        r.append(last_part)

    return r


def sort_tree_by_hash(t: ImmutableTree, sortable_nodes: Set[str]) -> ImmutableTree:
    if not isinstance(t, ImmutableTree) and not isinstance(t, str):
        raise ValueError("Got an unexpected type!")

    if isinstance(t, str):
        return t

    if t.label() in sortable_nodes:
        return ImmutableTree(t.label(),
                             children=sorted((sort_tree_by_hash(c, sortable_nodes) for c in t), key=lambda x: hash(x)))
    else:
        return ImmutableTree(t.label(), children=[sort_tree_by_hash(c, sortable_nodes) for c in t])


from typing import List, Dict

import nltk

class FunqlAcc:
    def __init__(self,  arities: Dict[str, int], sortable_nodes = ("intersection", "or"), format="reduced"):
        self.sortable_nodes = set(sortable_nodes)
        self.reset()
        self.arities = arities
        self.format = format

        assert self.format in ["lisp", "reduced"]

    def reset(self):
        self.correct = 0
        self.total = 0

    def get_metric(self, reset: bool) -> Dict[str, float]:
        if self.total == 0:
            return {"tree_acc": 0}
        a = self.correct / self.total
        if reset:
            self.reset()

        return {"tree_acc": a}

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]]) -> None:
        assert len(predictions) == len(gold)

        self.total += len(gold)

        for p, g in zip(predictions, gold):
            # For BART, which outputs strings instead of lists of tokens:
            p = p.strip()
            g = g.strip()

            if self.format == "lisp":
                try:
                    g_t = ImmutableTree.fromstring(g)
                except Exception:
                    print("Warning, could not convert gold sequence into tree", g)
                    continue
                try:
                    p_t = ImmutableTree.fromstring(p)
                except Exception:
                    continue

            elif self.format == "reduced":
                try:
                    g_t = reconstruct_tree_with_partial_brackets(g.split(" "), self.arities)
                except Exception as ex:
                    print("Warning, could not convert gold sequence into tree", g)
                    continue
                try:
                    p_t = reconstruct_tree_with_partial_brackets(p.split(" "), self.arities)
                except Exception:
                    continue

            self.correct += sort_tree_by_hash(nltk.ImmutableTree.convert(g_t), self.sortable_nodes) == \
                            sort_tree_by_hash(nltk.ImmutableTree.convert(p_t), self.sortable_nodes)



atis_arities = dict({('_stop_1', 1), ('argmax_capacity', 1), ('_fare_2', 1), ('_year_2', 1), ('year', 1), ('_to_2', 1), ('_economy', 1),
                ('_named_1', 1), ('days_code', 1), ('_rapid_transit', 1), ('_oneway', 1), ('month', 1), ('meal_description', 1),
                ('_<_departure_time_2', 1), ('_turboprop', 1), ('_booking_class_2', 1), ('_flight_aircraft', 1),
                ('_stops', 1), ('argmax_count', 1), ('_booking_class_1', 1), ('_booking_class:_t', 1), ('manufacturer', 1),
                ('_flight_number_2', 1), ('_round_trip', 1), ('_jet', 1), ('_fare_time', 1), ('fare_basis_code', 1), ('argmin_fare', 1),
                ('_tomorrow', 1), ('_flight_airline', 1), ('answer', 1), ('_minimum_connection_time', 1), ('_next_days_2', 1),
                ('_aircraft_basis_type_2', 1), ('argmin_miles_distant_2', 1), ('_limousine', 1), ('_fare', 1), ('_arrival_time', 1),
                ('_stop_2', 1), ('_nonstop', 1), ('_daily', 1), ('_aircraft_1', 1), ('_aircraft', 1), ('_services', 2), ('class_description', 1),
                ('argmax_stops', 1), ('_flight', 1), ('_stop_arrival_time', 1), ('argmin_time_elapsed', 1), ('day_period', 1), ('_from_2', 1),
                ('_loc:_t_1', 1), ('_time_zone_code', 1), ('_airport_1', 1), ('city_name', 1), ('time', 1), ('_time_elapsed', 1),
                ('_connecting', 1), ('_fare_basis_code', 1), ('_approx_return_time_2', 1), ('_meal_code_2', 1),
                ('day_number', 1), ('argmin_stops', 1), ('argmin_departure_time', 1), ('_from_1', 1),
                ('_airline_name', 1), ('_day_after_tomorrow', 1), ('_city', 1),
                ('_airport', 1), ('_month_2', 1), ('_ground_transport', 1), ('argmax_fare', 1),
                ('_restriction_code', 1), ('_to_1', 1), ('_meal', 1), ('_has_stops', 1),
                ('_>_arrival_time_2', 1), ('_<_fare_2', 1), ('_abbrev', 1), ('_days_from_today_2', 1),
                ('_overnight', 1), ('_during_day_2', 1), ('_airline', 1), ('airport_code', 1), ('_meal_code', 1),
                ('airline_code', 1), ('_after_day_2', 1), ('sum_stops', 1), ('_discounted', 1), ('_today', 1),
                ('_before_day_2', 1), ('_month_return_2', 1), ('_class_type_2', 1), ('_flight_number', 1),
                ('sum_capacity', 1), ('_day_number_return_2', 1), ('_min', 1), ('_time_elapsed_2', 1),
                ('_class_of_service', 1), ('_rental_car', 1), ('_max', 1), ('_fare_basis_code_2', 1),
                ('_aircraft_2', 1), ('_miles_distant', 2), ('_from_airport_2', 1),
                ('_air_taxi_operation', 1), ('_tomorrow_arrival', 1), ('_day_arrival_2', 1), ('_<_arrival_time_2', 1),
                ('aircraft_code', 1), ('_loc:_t_2', 1), ('flight_number', 1), ('_services_1', 1), ('count', 1),
                ('_airline_1', 1), ('_day_2', 1), ('_meal_2', 1), ('_to_city_2', 1), ('_capacity', 1),
                ('argmax_arrival_time', 1), ('_departure_time', 1), ('_taxi', 1), ('argmax_departure_time', 1),
                ('_during_day_arrival_2', 1), ('_approx_arrival_time_2', 1), ('_day_return_2', 1), ('_>_stops_2', 1),
                ('_manufacturer_2', 1), ('state_name', 1), ('day', 1), ('basis_type', 1), ('_departure_time_2', 1),
                ('_tonight', 1), ('_weekday', 1), ('_has_meal', 1), ('_day_number_arrival_2', 1), ('dollar', 1),
                ('_>_capacity_2', 1), ('argmin_capacity', 1), ('argmin_arrival_time', 1), ('_ground_fare', 1),
                ('hour', 1), ('_flight_fare', 1), ('not', 1), ('_services_2', 1), ('integer', 1),
                ('_month_arrival_2', 1), ('_equals', 2), ('_airline_2', 1), ('_day_number_2', 1),
                ('meal_code', 1), ('_minutes_distant', 1), ('_stops_2', 1), ('_>_departure_time_2', 1),
                ('_arrival_time_2', 1), ('_approx_departure_time_2', 1)})


class AtisAcc(FunqlAcc):

    def __init__(self, **kwargs):
        super().__init__(atis_arities, sortable_nodes = ("intersection", "or"), **kwargs)



class LevenstheinMetric:

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_instances = 0
        self.total_distance = 0
        self.correct_length = 0
        self.correct = 0

    def get_metric(self, reset: bool) -> Dict[str, float]:
        if self.total_instances == 0:
            return {"edit_dist": 0, "acc": 0, "per": 0}
        r = self.total_distance / self.total_instances
        acc = self.correct / self.total_instances
        per = self.total_distance / self.correct_length
        if reset:
            self.reset()
        return {"edit_dist": r, "acc": acc, "per": per}

    def add_instances(self, predictions: List[List[str]], gold: List[List[str]]) -> None:
        assert len(predictions) == len(gold)

        for p, g in zip(predictions, gold):
            if p != g:
               print("Gold", g)
               print("Pred", p)
               print("---")
            self.total_instances += 1
            self.total_distance += Levenshtein.distance(p, g)
            self.correct += p == g
            self.correct_length += len(g)


def create_atis_eval(format="reduced", **kwargs):
    return lambda *args, **kwargs2: atis_eval_match(*args, format=format, **(kwargs2 | kwargs))


def atis_eval_match(model, tokenizer, dataloader, format="reduced", logger=None, **kwargs):
  model.eval()
  atis_eval = AtisAcc(format=format)
  lev_metric = LevenstheinMetric()
  for test_batch in dataloader:
    test_batch = {k: v.to(model.device) for k,v in test_batch.items()}
    test_batch_inputs = dict(test_batch)
    del test_batch_inputs["labels"]
    r = tokenizer.batch_decode(model.generate(**test_batch_inputs, max_new_tokens=test_batch["labels"].shape[1]+2,
                                              early_stopping="never", num_beams=1, no_repeat_ngram_size=0), skip_special_tokens=True)
    gold = tokenizer.batch_decode((100 + tokenizer.eos_token_id) *(test_batch["labels"] == -100) + test_batch["labels"], skip_special_tokens=True) # replace -100 by eos token id, which will be skipped.

    if logger is not None:
        logger.log_output([{"prediction": pi, "gold": gi, "input": ii}
                           for pi, gi, ii in zip(r, gold, tokenizer.batch_decode(test_batch["input_ids"], skip_special_tokens=True))])

    atis_eval.add_instances(r, gold)
    lev_metric.add_instances(r, gold)

  return atis_eval.get_metric(True) | lev_metric.get_metric(True)





if __name__ == "__main__":
    # x = "answer _stop_1 _flight ( intersection _airline_2 airline_code dl _flight_number_2 flight_number 838 _from_2 city_name san_francisco _to_2 city_name atlanta )"
    x = "answer _stop_1 _flight ( intersection _airline_2 airline_code dl ( intersection _airline_2 airline_code dl _airline_2 airline_code BLA ) _flight_number_2 flight_number 838 _from_2 city_name san_francisco _to_2 city_name atlanta )"
    x2 = "answer _stop_1 _flight ( intersection _from_2 city_name san_francisco _airline_2 airline_code dl ( intersection _airline_2 airline_code BLA _airline_2 airline_code dl ) _flight_number_2 flight_number 838 _to_2 city_name atlanta )"
    x = x.split(" ")
    x2 = x2.split(" ")
    arities = {"answer": 1, "_stop_1": 1,
               "_flight": 1,"_airline_2": 1, "airline_code": 1,
               "_flight_number_2": 1,"flight_number": 1, "_from_2": 1,"city_name": 1, "_to_2": 1
               }
    t = reconstruct_tree_with_partial_brackets(x, arities)
    t2 = reconstruct_tree_with_partial_brackets(x2, arities)

    t_sort = sort_tree_by_hash(ImmutableTree.convert(t), {"intersection"})
    t2_sort = sort_tree_by_hash(ImmutableTree.convert(t2), {"intersection"})

    print(t_sort)
    print(t2_sort)
    print(t_sort == t2_sort)


