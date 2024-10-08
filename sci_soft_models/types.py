#!/usr/bin/env python

from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin

###############################################################################


@dataclass
class ModelDetails(DataClassJsonMixin):
    name: str
    version: str
