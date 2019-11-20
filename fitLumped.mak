#!/usr/bin/make -f
all: $(shell echo p{0..3323}) Min3data_2iso_1105_Lys.h5

Min3data_2iso_1105_Lys.h5:
	python fitLysLumped.py

p%: Min3data_2iso_1105_Lys.h5
	python fitPLumped.py $(subst p,,$@)