#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = u"Frank Jing"

#my_module.py
print('from the my_module.py')

money=1000

def read1():
    print('my_module->read1->money',money)

def read2():
    print('my_module->read2 calling read1')
    read1()

def change():
    global money
    money=0

def somefunc(a: int = ..., b: str = ...) -> list: ...



def main():
    pass


if __name__ == '__main__':
    main()