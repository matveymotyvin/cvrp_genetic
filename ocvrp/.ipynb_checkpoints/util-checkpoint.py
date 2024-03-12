import collections
import math
from json import JSONEncoder
from typing import Tuple, List, Union


class Building:

    def __init__(self, node: int, x: float, y: float, quant: int):
        self._x = x
        self._y = y
        self._quant = quant
        self._node = node

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        self._x = x

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        self._y = y

    @property
    def quant(self) -> int:
        return self._quant

    @quant.setter
    def quant(self, capacity: int) -> None:
        self._quant = capacity

    @property
    def node(self) -> int:
        return self._node

    @node.setter
    def node(self, ident: int) -> None:
        self._node = ident

    @staticmethod
    def distance(b1: 'Building', b2: 'Building'):
        return round(math.sqrt(((b1.x - b2.x) ** 2) + ((b1.y - b2.y) ** 2)))

    def __str__(self):
        return f"Node: {self.node}, x: {self.x}, y: {self.y}, quant: {self.quant}"

    def __repr__(self):
        return f"util.Building<node: {self.node}, x: {self.x}, y: {self.y}, quant: {self.quant}>"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._node == other.node
        return False

    def __key(self) -> Tuple:
        return self._node, self._x, self._y, self._quant

    def __hash__(self):
        return hash(self.__key())


class CVRPEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Building):
            return o.__dict__


class Individual(collections.abc.Sequence):

    def __init__(self, genes, fitness: Union[None, float]):
        self._genes = genes
        self._fitness = fitness

    @property
    def genes(self) -> List[Building]:
        return self._genes

    @genes.setter
    def genes(self, genes: List[Building]) -> None:
        self._genes = genes

    @property
    def fitness(self) -> Union[float, None]:
        return self._fitness

    @fitness.setter
    def fitness(self, fitness: float) -> None:
        self._fitness = fitness

    def __key(self) -> Tuple:
        return (self._fitness,)

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return f"Genes: {self._genes},\nfitness: {self._fitness}"

    def __repr__(self):
        return f"util.Individual<genes: {self._genes}, fitness: {self._fitness}>"

    def __iter__(self):
        for g in self._genes:
            yield g

    def __contains__(self, item):
        return item in self._genes

    def __getitem__(self, item):
        return self._genes[item]

    def __setitem__(self, key, value):
        self._genes[key] = value

    def __delitem__(self, key):
        self.__delattr__(key)

    def __len__(self):
        return len(self._genes)

    def __eq__(self, other):
        return self._fitness == other.fitness

    def __ne__(self, other):
        return self._fitness != other.fitness

    def __lt__(self, other):
        return self._fitness < other.fitness

    def __le__(self, other):
        return self._fitness <= other.fitness

    def __ge__(self, other):
        return self._fitness >= other.fitness

    def __gt__(self, other):
        return self._fitness > other.fitness

    def __radd__(self, other):
        return other + self._fitness


class Vehicle:
    def __init__(self, capacity: int):
        self._capacity = capacity

    @property
    def capacity(self) -> int:
        return self._capacity

    @capacity.setter
    def capacity(self, capacity: int) -> None:
        self._capacity = capacity

class OCVRPParser:
    def __init__(self, filename):
        
        if not filename.endswith(".ocvrp"):
            raise SyntaxError("File is not of .ocvrp type")

        self.f = open(filename, "r")
        self._values = {}
        self._headers = ("NAME", "COMMENTS", "DIM", "CAPACITY", "OPTIMAL")
        self._num_headers = ("DIM", "CAPACITY", "OPTIMAL")

    class __OCVRPParserStrategy:

        def __init__(self, parser):
            self.__parser = parser

        def get_ps_depot(self):
            return self.__parser._values["DEPOT"]

        def get_ps_buildings(self):
            return self.__parser._values["BUILDINGS"]

        def get_ps_name(self):
            return self.__parser._values["NAME"]

        def get_ps_comments(self):
            return self.__parser._values["COMMENTS"] if "COMMENTS" in self.__parser._values else None

        def get_ps_dim(self):
            return self.__parser._values["DIM"] - 1

        def get_ps_capacity(self):
            return self.__parser._values["CAPACITY"]

        def get_ps_optimal(self):
            return self.__parser._values["OPTIMAL"]

    def parse(self):
        
        lines = self.f.readlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line not in ('\n', '\r\n'):
                try:
                    ln = line.split(":")
                    ln0 = ln[0].upper()
                    ln1 = ln[1]

                    # If the header is the node, we need to load all the nodes underneath the header
                    if ln0 == 'NODES':
                        idx = self._grab_buildings(idx + 1, lines)
                    else:
                        if self._is_number(ln1) and ln0 in self._num_headers:
                            # Load the header as an integer if it's numeric
                            self._values[ln0] = int(ln1)
                        else:
                            self._values[ln0] = ln1.replace("\n", "").strip()
                except Exception as e:
                    raise SyntaxError("File is not formatted properly", e)
            idx += 1

        self.f.close()
        return self.__OCVRPParserStrategy(self)

    def _grab_buildings(self, curr, lines):
        ll = len(lines)
        buildings = []
        ctr = 0

        while curr < ll:
            line = lines[curr]
            ls = line.split()

            ident, x, y, quant = ls
            h = Building(int(ident), float(x), float(y), int(quant))

            # First node is always DEPOT
            if ctr == 0:
                self._values["DEPOT"] = h
            else:
                buildings.append(h)

            # Check if EOF or if the next line is numeric (if not, indicates that node parsing is done)
            if curr < ll - 1:
                next_num = lines[curr + 1].split()
                if len(next_num) == 0 or not self._is_number(next_num[0]):
                    break
            else:
                break

            ctr += 1
            curr += 1

        self._values["BUILDINGS"] = buildings
        return curr

    @staticmethod
    def _is_number(num: str):
        
        try:
            float(num)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(num)
            return True
        except (TypeError, ValueError):
            pass

        return False


