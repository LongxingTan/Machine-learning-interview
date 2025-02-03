# 停车场设计


## 代码

```
public abstract class Vehicle {
    protected String licensePlate;
    protected VehicleSize size;

    public Vehicle(String licensePlate, VehicleSize size) {
        this.licensePlate = licensePlate;
        this.size = size;
    }

    public String getLicensePlate() {
        return licensePlate;
    }

    public VehicleSize getSize() {
        return size;
    }

    public abstract String getType();
}
```

```java
public enum VehicleSize {
    MOTORCYCLE,
    COMPACT,
    REGULAR,
    LARGE;
}
```

```java
public class Car extends Vehicle {
    public Car(String licensePlate) {
        super(licensePlate, VehicleSize.REGULAR);
    }

    @Override
    public String getType() {
        return "Car";
    }
}

public class Truck extends Vehicle {
    public Truck(String licensePlate) {
        super(licensePlate, VehicleSize.LARGE);
    }

    @Override
    public String getType() {
        return "Truck";
    }
}

public class Motorcycle extends Vehicle {
    public Motorcycle(String licensePlate) {
        super(licensePlate, VehicleSize.MOTORCYCLE);
    }

    @Override
    public String getType() {
        return "Motorcycle";
    }
}
```

```java
public class ParkingSpot {
    private int spotNumber;
    private VehicleSize size;
    private Vehicle parkedVehicle;

    public ParkingSpot(int spotNumber, VehicleSize size) {
        this.spotNumber = spotNumber;
        this.size = size;
        this.parkedVehicle = null;  // Empty spot initially
    }

    public boolean isOccupied() {
        return parkedVehicle != null;
    }

    public boolean canFit(Vehicle vehicle) {
        return !isOccupied() && vehicle.getSize().ordinal() <= size.ordinal();
    }

    public boolean park(Vehicle vehicle) {
        if (canFit(vehicle)) {
            parkedVehicle = vehicle;
            return true;
        }
        return false;
    }

    public Vehicle removeVehicle() {
        Vehicle temp = parkedVehicle;
        parkedVehicle = null;
        return temp;
    }

    public int getSpotNumber() {
        return spotNumber;
    }

    public Vehicle getParkedVehicle() {
        return parkedVehicle;
    }
}
```

```java
import java.util.ArrayList;
import java.util.List;

public class ParkingLot {
    private List<ParkingSpot> spots;

    public ParkingLot(int totalSpots) {
        spots = new ArrayList<>();
        // Assume alternating spot types for simplicity
        for (int i = 1; i <= totalSpots; i++) {
            VehicleSize size = (i % 3 == 0) ? VehicleSize.LARGE :
                                (i % 2 == 0) ? VehicleSize.REGULAR : VehicleSize.COMPACT;
            spots.add(new ParkingSpot(i, size));
        }
    }

    public boolean parkVehicle(Vehicle vehicle) {
        for (ParkingSpot spot : spots) {
            if (spot.canFit(vehicle)) {
                return spot.park(vehicle);
            }
        }
        return false; // No suitable spot
    }

    public Vehicle removeVehicle(String licensePlate) {
        for (ParkingSpot spot : spots) {
            if (spot.isOccupied() && spot.getParkedVehicle().getLicensePlate().equals(licensePlate)) {
                return spot.removeVehicle();
            }
        }
        return null; // Vehicle not found
    }

    public int getAvailableSpots() {
        int availableCount = 0;
        for (ParkingSpot spot : spots) {
            if (!spot.isOccupied()) {
                availableCount++;
            }
        }
        return availableCount;
    }

    public List<ParkingSpot> getAllSpots() {
        return spots;
    }
}
```

```java
public class ParkingLotManagementSystem {
    private ParkingLot parkingLot;

    public ParkingLotManagementSystem(int totalSpots) {
        parkingLot = new ParkingLot(totalSpots);
    }

    public boolean parkVehicle(Vehicle vehicle) {
        return parkingLot.parkVehicle(vehicle);
    }

    public Vehicle removeVehicle(String licensePlate) {
        return parkingLot.removeVehicle(licensePlate);
    }

    public int getAvailableSpots() {
        return parkingLot.getAvailableSpots();
    }

    public void displayParkingLotStatus() {
        System.out.println("Parking Lot Status:");
        for (ParkingSpot spot : parkingLot.getAllSpots()) {
            String status = spot.isOccupied() ? "Occupied by " + spot.getParkedVehicle().getType() :
                    "Available";
            System.out.println("Spot " + spot.getSpotNumber() + ": " + status);
        }
    }
}
```
