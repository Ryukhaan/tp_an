# -*- coding: utf-8 -*-
import math


class SonarData:
    def __init__(self, data):
        self.type = data[-1]
        self.origin = data[0][:-1] # le '-1' est pour enlever le caratère de retour chariot.
        self.attributes = data[1:-1]

    def calc_distance(self, other):
        if isinstance(other, SonarData):
            res = 0
            for i in range(len(self.attributes)):
                res += math.fabs(self.get_attribute(i) - other.get_attribute(i))
            return res
        else:
            raise TypeError("'other' object must be an instance of SonarData.")

    def get_sonar_class(self):
        return self.type

    def is_same_sonar(self, other):
        if isinstance(other, SonarData):
            return self.get_sonar_class() == other.get_sonar_class()
        elif isinstance(other, str):
            return self.get_sonar_class() == other
        else:
            raise Exception("'other' must be a string or a SonarData Object.")

    def __str__(self):
        return "Sonar signal:{0.type}, base: {0.origin}, value: {0.attributes}".format(self)

    def get_attribute(self, i):
        return float(self.attributes[i])