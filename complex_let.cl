class Main {
  main() : Int {
    let x : Int <- 
      if 1 < 2 then
        let y : Int <- 3 in y * 2
      else
        while 1 < 2 loop 5 pool
      fi
    in x
  };
};
