#!/usr/bin/python

import sys, string, os, time, re
import random

def main():

  if len(sys.argv) < 5:
    print "usage: ./RandomHits.py Evts Roads Hits outFile [type]\n"  
    print "   Evts      number of events to generate\n"
    print "   Roads     number of roads per event to generate\n"
    print "   Hits      number of hits per layer to generate\n"
    print "   outFile   output file name\n"
    print "   type      [optional] if 1 uses random numbers for #roads and #hits\n"
    return -1

  # Default values 
  fixed = 1
  NSVXhits = 5
  NSVThits = 6
  logfile = "hitsDecoded.log"

  # read arguments
  NEvts = int(sys.argv[1]) 
  NRoads = int(sys.argv[2])  
  NHitsPerLayer = int(sys.argv[3]) 

  out = open(sys.argv[4],"w")
  log = open(logfile,"w")

  if len(sys.argv) == 6 :
    fixed = 0

  print "Generating random hits for SVT test (See %s for log) \n" % logfile
  if fixed:
    print "Generating %d events, with %d roads, with %d hits per layer \n" % (NEvts, NRoads, NHitsPerLayer)
  else:
    print "Generating %d events, with at most %d roads, with at most %d hits per layer \n" % (NEvts, NRoads, NHitsPerLayer)

  numroad = 0
  maxroad = 0
  numcomb = 0
  maxcomb = 0

  for nevt in range(0,NEvts):
    log.write("***** Event  #%d \n" % nevt)
    if fixed:
      maxr = NRoads
    else:
      maxr = random.randint(0, NRoads)
    if maxr > maxroad: maxroad = maxr
    numroad += maxr
    for h in range(0, maxr): 
      log.write("\t Road   #%d \n" % h)
      comb = 1
      for i in range(0,NSVThits): 
        log.write("\t\t Layer num #%d \n" % i) 
        if fixed:
          maxhl = NHitsPerLayer
        else:
          maxhl = random.randint(1, NHitsPerLayer)
        if i: comb *= maxhl  
        for j in range(0,maxhl):
          log.write("\t\t\t Hit num #%d \n" % j)
          if i < NSVXhits:
            # 15 bits for hit coordinate (bit 0 - 14)
            hit = random.randint(0, 0x7fff)
            log.write("\t\t\t\t value: %.6x \n" % hit)  
            # 3 bits for the barrel (random number) bit 15-17
            zbarrel = random.randint(0,0x3)
            log.write("\t\t\t\t zbarrel: %.6x \n" % zbarrel)
            hit +=  (zbarrel << 15)  
            # 3 bits for the layer, bit 18-20
            layer = i
            log.write("\t\t\t\t layer: %.6x \n" % layer) 
            hit +=  (layer << 18)
            if layer != 0:
              out.write("%.6x\n" % hit)
          elif i == NSVXhits:
            # XFT information for the AM, 18 bits (0-17)
            hit = 0x1feed
            log.write("\t\t\t\t XFT for AM: %.6x\n" % hit) 
            # layer (XFT), 3 bits, (18-20)
            layer = i
            log.write("\t\t\t\t layer(XFT): %.6x\n" % layer)
            hit += (layer << 18)
            out.write("%.6x\n" % hit)
            # phi, 12 bits (0-11)
            phi = random.randint(0,0xfff)
            hit = phi
            log.write("\t\t\t\t phi: %.6x\n" % phi)
            # curvature, 7 bits (last bit is the sign), 12-18
            curv = random.randint(0,0x3f)
            log.write("\t\t\t\t curvature: %.6x\n" % curv)
            hit += (curv << 12)
            out.write("%.6x\n" % hit)

      # AM road ID, 21 bits, 0-20
      amroadID = random.randint(0,0x1fffff)
      hit = amroadID
      log.write("\t\t amroadID: %.6x\n" % amroadID)
      # EP = 1
      hit += (1<<21)
      log.write("\t\t EP: %.6x\n" % 1)
      out.write("%.6x\n" % hit)
  
      if comb > maxcomb: maxcomb = comb
      numcomb +=comb

    # last word (must contain EE = 1)
    # for the moment, 21 bits random
    lastword = random.randint(0,0x1fffff)
    hit = lastword
    log.write("\tLastword: %.6x\n" % lastword)
    # EP = 1 (bit 21)
    hit += (1<<21)
    log.write("\tEP: %.6x\n" % 1)
    # EE = 1 (bit 22)
    hit += (1<<22)
    log.write("\tEE: %.6x\n" % 1)
    out.write("%.6x\n" % hit)

  log.write("--------------------------------\n")
  log.write("Generate %d events\n" %NEvts) 
  log.write("Generate %d total road (with a maximum of %d over events)\n" %(numroad, maxroad)) 
  log.write("Generate %d total combination (with a maximum of %d over roads)\n" % (numcomb, maxcomb)) 

  print "We generate a total of %d roads (max: %d) and %d combinations (max: %d)\n" % (numroad, maxroad, numcomb, maxcomb)


if __name__=="__main__":
  main()
