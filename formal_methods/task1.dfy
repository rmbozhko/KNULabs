class CoffeeMachine {
    var isOn: bool
    var isPreparingDrink: bool
    var selectedDrink: string
    var isPaid: bool
    var isDrinkPrepared: bool
    var totalIncome: int
    var totalDrinksMade: int
    
    var coffeeAmount: int
    var milkAmount: int
    var waterAmount: int
    var sugarAmount: int
    
    var isTerminalWorking: bool

    constructor(initialCoffee: int, initialMilk: int, initialWater: int, initialSugar: int)
        requires initialCoffee >= 0
        requires initialMilk >= 0
        requires initialWater >= 0
        requires initialSugar >= 0
        ensures coffeeAmount == initialCoffee
        ensures milkAmount == initialMilk
        ensures waterAmount == initialWater
        ensures sugarAmount == initialSugar
        ensures isOn == false
        ensures isPreparingDrink == false
        ensures selectedDrink == ""
        ensures isPaid == false
        ensures isDrinkPrepared == false
        ensures totalIncome == 0
        ensures totalDrinksMade == 0
        ensures isTerminalWorking == true
    {
        coffeeAmount := initialCoffee;
        milkAmount := initialMilk;
        waterAmount := initialWater;
        sugarAmount := initialSugar;
        isOn := false;
        isPreparingDrink := false;
        selectedDrink := "";
        isPaid := false;
        isDrinkPrepared := false;
        totalIncome := 0;
        totalDrinksMade := 0;
        isTerminalWorking := true;
    }
    
    method PowerToggle() 
        modifies this
        ensures old(isOn) ==> !isOn
        ensures !old(isOn) ==> isOn
    {
        isOn := !isOn;
        if (!isOn) {
            isPreparingDrink := false;
            selectedDrink := "";
            isPaid := false;
            isDrinkPrepared := false;
        }
    }
    
    method HasEnoughIngredients(drinkType: string) returns (hasEnough: bool)
    {
        if (drinkType == "espresso") {
            hasEnough := coffeeAmount >= 7 && waterAmount >= 30;
        } else if (drinkType == "cappuccino") {
            hasEnough := coffeeAmount >= 7 && milkAmount >= 70 && waterAmount >= 30;
        } else if (drinkType == "latte") {
            hasEnough := coffeeAmount >= 7 && milkAmount >= 120 && waterAmount >= 30;
        } else if (drinkType == "americano") {
            hasEnough := coffeeAmount >= 7 && waterAmount >= 100;
        } else {
            hasEnough := false;
        }
        
        return hasEnough;
    }
    
    method SelectDrink(drinkName: string) returns (success: bool)
        modifies this
        ensures !old(isOn) ==> !success
        ensures old(isPreparingDrink) ==> !success
        ensures success ==> selectedDrink == drinkName
    {
        if (!isOn || isPreparingDrink) {
            success := false;
            return success;
        }

        var hasIngredients := HasEnoughIngredients(drinkName);
        if (!hasIngredients) {
            success := false;
            return success;
        }

        selectedDrink := drinkName;
        success := true;
        return success;
    }
    
    method PayForDrink(drinkPrice: int) returns (success: bool)
        modifies this
        requires drinkPrice > 0
        ensures selectedDrink == "" ==> !success
        ensures !isTerminalWorking ==> !success
        ensures success ==> isPaid
        ensures success ==> totalIncome == old(totalIncome) + drinkPrice
        ensures success ==> totalDrinksMade == old(totalDrinksMade) + 1
    {
        if (selectedDrink == "") {
            success := false;
            return success;
        }
        
        if (!isTerminalWorking) {
            success := false;
            return success;
        }
        
        isPaid := true;
        totalIncome := totalIncome + drinkPrice;
        totalDrinksMade := totalDrinksMade + 1;
        success := true;
        return success;
    }
    
    method PrepareDrink() returns (success: bool)
        modifies this
        ensures !isPaid ==> !success
        ensures success ==> isDrinkPrepared
        ensures success ==> isPreparingDrink
    {
        if (!isPaid) {
            success := false;
            return success;
        }

        isPreparingDrink := true;
        isDrinkPrepared := true;
        success := true;
        return success;
    }
    
    method DispenseDrink() returns (success: bool)
        modifies this
        ensures !isDrinkPrepared ==> !success
        ensures success ==> !isDrinkPrepared
        ensures success ==> !isPreparingDrink
        ensures success ==> !isPaid
        ensures success ==> selectedDrink == ""
    {
        if (!isDrinkPrepared) {
            success := false;
            return success;
        }
        
        if (selectedDrink == "espresso") {
            coffeeAmount := coffeeAmount - 7;
            waterAmount := waterAmount - 30;
        } else if (selectedDrink == "cappuccino") {
            coffeeAmount := coffeeAmount - 7;
            milkAmount := milkAmount - 70;
            waterAmount := waterAmount - 30;
        } else if (selectedDrink == "latte") {
            coffeeAmount := coffeeAmount - 7;
            milkAmount := milkAmount - 120;
            waterAmount := waterAmount - 30;
        } else if (selectedDrink == "americano") {
            coffeeAmount := coffeeAmount - 7;
            waterAmount := waterAmount - 100;
        }
        
        isDrinkPrepared := false;
        isPreparingDrink := false;
        isPaid := false;
        selectedDrink := "";
        
        success := true;
        return success;
    }

    method ToggleTerminal()
        modifies this
        ensures isTerminalWorking == !old(isTerminalWorking)
    {
        isTerminalWorking := !isTerminalWorking;
    }
}

method Main()
{
    var machine := new CoffeeMachine(500, 1000, 2000, 300);
    
    machine.PowerToggle();
    print "Machine powered on\n";
    
    var selectionSuccess := machine.SelectDrink("cappuccino");
    if (selectionSuccess) {
        print "Selected drink: cappuccino\n";
    } else {
        print "Failed to select drink - not enough ingredients or machine not ready\n";
    }
    
    var paymentSuccess := machine.PayForDrink(35);
    if (paymentSuccess) {
        print "Payment successful. Total income: ", machine.totalIncome, "\n";
    } else {
        print "Payment failed\n";
    }
    
    var preparationSuccess := machine.PrepareDrink();
    if (preparationSuccess) {
        print "Drink preparation started\n";
    } else {
        print "Failed to prepare drink\n";
    }
    
    var dispensingSuccess := machine.DispenseDrink();
    if (dispensingSuccess) {
        print "Drink dispensed. Remaining ingredients:\n";
        print "Coffee: ", machine.coffeeAmount, "g\n";
        print "Milk: ", machine.milkAmount, "ml\n";
        print "Water: ", machine.waterAmount, "ml\n";
        print "Sugar: ", machine.sugarAmount, "g\n";
    } else {
        print "Failed to dispense drink\n";
    }

    print "\nTotal drinks made: ", machine.totalDrinksMade, "\n";
    print "Total income: ", machine.totalIncome, "\n";
}