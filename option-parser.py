#!/usr/bin/python
import os.path
from optparse import OptionParser


if __name__ == "__main__": 
    parses = OptionParser()
    parses.add_option("-c", "--chessboards", dest="chessboards", 
                      help="use chessboards in file CHESSBOARDS", 
                      metavar="CHESSBOARDS", 
                      default="chessboards.txt")
    parses.add_option("-n", "--numboards", dest="n", 
                      help="use N chessboards for calibration", 
                      metavar="N", default="8")
    parses.add_option("-d", "--dir", dest="dir", 
                      help="store calibration in DIR", 
                      metavar="DIR", default="calib")
    (options, args) = parses.parse_args()

