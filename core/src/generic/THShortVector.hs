{-# LANGUAGE ForeignFunctionInterface #-}

module THShortVector (
    c_THShortVector_fill,
    c_THShortVector_cadd,
    c_THShortVector_adds,
    c_THShortVector_cmul,
    c_THShortVector_muls,
    c_THShortVector_cdiv,
    c_THShortVector_divs,
    c_THShortVector_copy,
    c_THShortVector_vectorDispatchInit,
    p_THShortVector_fill,
    p_THShortVector_cadd,
    p_THShortVector_adds,
    p_THShortVector_cmul,
    p_THShortVector_muls,
    p_THShortVector_cdiv,
    p_THShortVector_divs,
    p_THShortVector_copy,
    p_THShortVector_vectorDispatchInit) where

import Foreign
import Foreign.C.Types
import THTypes

-- |c_THShortVector_fill : x c n -> void
foreign import ccall unsafe "THVector.h THShortVector_fill"
  c_THShortVector_fill :: Ptr CShort -> CShort -> CPtrdiff -> IO ()

-- |c_THShortVector_cadd : z x y c n -> void
foreign import ccall unsafe "THVector.h THShortVector_cadd"
  c_THShortVector_cadd :: Ptr CShort -> Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ()

-- |c_THShortVector_adds : y x c n -> void
foreign import ccall unsafe "THVector.h THShortVector_adds"
  c_THShortVector_adds :: Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ()

-- |c_THShortVector_cmul : z x y n -> void
foreign import ccall unsafe "THVector.h THShortVector_cmul"
  c_THShortVector_cmul :: Ptr CShort -> Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ()

-- |c_THShortVector_muls : y x c n -> void
foreign import ccall unsafe "THVector.h THShortVector_muls"
  c_THShortVector_muls :: Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ()

-- |c_THShortVector_cdiv : z x y n -> void
foreign import ccall unsafe "THVector.h THShortVector_cdiv"
  c_THShortVector_cdiv :: Ptr CShort -> Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ()

-- |c_THShortVector_divs : y x c n -> void
foreign import ccall unsafe "THVector.h THShortVector_divs"
  c_THShortVector_divs :: Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ()

-- |c_THShortVector_copy : y x n -> void
foreign import ccall unsafe "THVector.h THShortVector_copy"
  c_THShortVector_copy :: Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ()

-- |c_THShortVector_vectorDispatchInit :  -> void
foreign import ccall unsafe "THVector.h THShortVector_vectorDispatchInit"
  c_THShortVector_vectorDispatchInit :: IO ()

-- |p_THShortVector_fill : Pointer to x c n -> void
foreign import ccall unsafe "THVector.h &THShortVector_fill"
  p_THShortVector_fill :: FunPtr (Ptr CShort -> CShort -> CPtrdiff -> IO ())

-- |p_THShortVector_cadd : Pointer to z x y c n -> void
foreign import ccall unsafe "THVector.h &THShortVector_cadd"
  p_THShortVector_cadd :: FunPtr (Ptr CShort -> Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ())

-- |p_THShortVector_adds : Pointer to y x c n -> void
foreign import ccall unsafe "THVector.h &THShortVector_adds"
  p_THShortVector_adds :: FunPtr (Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ())

-- |p_THShortVector_cmul : Pointer to z x y n -> void
foreign import ccall unsafe "THVector.h &THShortVector_cmul"
  p_THShortVector_cmul :: FunPtr (Ptr CShort -> Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ())

-- |p_THShortVector_muls : Pointer to y x c n -> void
foreign import ccall unsafe "THVector.h &THShortVector_muls"
  p_THShortVector_muls :: FunPtr (Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ())

-- |p_THShortVector_cdiv : Pointer to z x y n -> void
foreign import ccall unsafe "THVector.h &THShortVector_cdiv"
  p_THShortVector_cdiv :: FunPtr (Ptr CShort -> Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ())

-- |p_THShortVector_divs : Pointer to y x c n -> void
foreign import ccall unsafe "THVector.h &THShortVector_divs"
  p_THShortVector_divs :: FunPtr (Ptr CShort -> Ptr CShort -> CShort -> CPtrdiff -> IO ())

-- |p_THShortVector_copy : Pointer to y x n -> void
foreign import ccall unsafe "THVector.h &THShortVector_copy"
  p_THShortVector_copy :: FunPtr (Ptr CShort -> Ptr CShort -> CPtrdiff -> IO ())

-- |p_THShortVector_vectorDispatchInit : Pointer to  -> void
foreign import ccall unsafe "THVector.h &THShortVector_vectorDispatchInit"
  p_THShortVector_vectorDispatchInit :: FunPtr (IO ())