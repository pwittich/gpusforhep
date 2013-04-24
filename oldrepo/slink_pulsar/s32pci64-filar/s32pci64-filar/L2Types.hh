//
// File: L2Types.hpp
// Purpose: L2 Trigger types include file
// Created: 18-MAR-1999 by Roger Moore
//
// Comments:
//     This header file defines the various variable types
//     used in the level 2 trigger.
//
// Revisions:
//
// 18-MAR-99 RWM:
//   Separated the generic include file into component pieces
//

#ifndef _L2UTILS_L2TYPES_HPP
#define _L2UTILS_L2TYPES_HPP

// KH
#define Linux
// KH


/** @name Type definitions for the various variable sizes */
//@{
/// byte sized integer variable
typedef unsigned char  byte;
/// 8 bit integer variable
typedef char  int8;
/// 16 bit integer variable
typedef short int16;
/// 32 bit integer variable
typedef int   int32;
/// 64 bit integer variable
#ifdef Linux
typedef long long int int64;
#else
typedef long  int64;
#endif
/// unsigned byte sized integer variable
typedef unsigned char  ubyte;
/// unsigned 8 bit integer variable
typedef unsigned char  uint8;



/// unsigned 16 bit integer variable
typedef unsigned short uint16;
/// unsigned 32 bit integer variable
typedef unsigned int   uint32;
/// unsigned 64 bit integer variable
#ifdef Linux
typedef unsigned long long int uint64;
#else
typedef unsigned long uint64;
#endif
/// Memory address variable
typedef unsigned long maddr;

#endif // _L2UTILS_L2TYPES_HPP
