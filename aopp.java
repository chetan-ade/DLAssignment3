class threadObj1 extends Thread {
    sharedObject sObj1, sObj2;    // sObj1 = ⊥, sObj2 = ⊥

    threadObj1(sharedObject so1, sharedObject so2) {
        sObj1 = so1;    // sObj1 = ⊥
        sObj2 = so2;    // sObj2 = ⊥
    }

    public void run() {
        sObj1.field = 2;                      // sObj1 = ⊥ (No Updates!) 
        sObj2.field = 1 + sObj1.field;        // sObj2 = ⊥ (No Updates!)
    }

    int identity(int m) {
        return m;
    }
}

class threadObj2 extends Thread {
    sharedObject sObj1, sObj2;    // sObj1 = ⊥, sObj2 = ⊥ (No Updates!)

    threadObj2(sharedObject so1, sharedObject so2) {
        sObj1 = so1;    // sObj1 = ⊥
        sObj2 = so2;    // sObj2 = ⊥
    }

    public void run() {
        sObj1.field = 3;                      // sObj1 = ⊥ (No Updates!)
        sObj2.field = 1 + sObj1.field;        // sObj2 = ⊥ (No Updates!)
    }

    int identity(int m) {
        return m;
    }
}

class aopp {
    public static void main(String args[]) {
        sharedObject so1 = new sharedObject();    // so1 = ⊥
        sharedObject so2 = new sharedObject();    // so2 = ⊥

        threadObj1 T1 = new threadObj1(so1, so2);
        threadObj2 T2 = new threadObj2(so1, so2);

        int a, b, c, s, p;  // a = ⊥, b = ⊥  // c = T, s = T, p = T
        s = 3;              // s = T -> 3                                       
        p = 4;              // p = T -> 4
        T1.start();  
        T2.start();
        
        c = T1.identity(3);   // c = T -> 3 -> ⊥ (as context insensitive)  
        c = T2.identity(4);

        a = T1.sObj1.field + 3;    // a = ⊥ (No Updates!)
        b = T2.sObj2.field + 4;    // b = ⊥ (No Updates!)
        c = s + p;                 // c = ⊥

        System.out.println(a + b + c);
    }
}

class sharedObject {
    int field;
}



