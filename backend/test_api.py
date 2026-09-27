import main

def test_backend_functions():
    print("=" * 60)
    print("Testing Backend API Functions & Database Queries (Phase 3)")
    print("=" * 60)

    # 1. Health Check
    health = main.health_check()
    print("\n1. Health Check:", health)
    assert health["status"] == "ok"

    # 2. Dashboard API
    print("\n2. Testing get_dashboard()...")
    dashboard = main.get_dashboard()
    print("Keys in Dashboard response:", list(dashboard.keys()))
    print("  - total_products:", dashboard["total_products"])
    print("  - total_inventory:", dashboard["total_inventory"])
    print("  - units_sold:", dashboard["units_sold"])
    print("  - inventory_health:", dashboard["inventory_health"])
    print("  - forecast_summary:", dashboard["forecast_summary"])
    print("  - low_stock_products:", dashboard["low_stock_products"])
    print("  - inventory_risk_summary:", dashboard["inventory_risk_summary"])
    print("  - category_distribution count:", len(dashboard["category_distribution"]))
    print("  - region_distribution count:", len(dashboard["region_distribution"]))
    
    assert dashboard["total_products"] > 0
    assert dashboard["total_inventory"] > 0
    assert "healthy" in dashboard["inventory_health"]

    # 3. Forecast API
    print("\n3. Testing get_forecast()...")
    forecast = main.get_forecast()
    print("Keys in Forecast response:", list(forecast.keys()))
    print("  - winning_model:", forecast["winning_model"])
    print("  - rmse:", forecast["rmse"])
    print("  - mae:", forecast["mae"])
    print("  - mape:", forecast["mape"])
    print("  - r2_score:", forecast["r2_score"])
    print("  - combined_forecast count:", len(forecast["combined_forecast"]))
    print("  - per_product_forecast count:", len(forecast["per_product_forecast"]))
    print("  - benchmark_results count:", len(forecast["benchmark_results"]))

    assert forecast["winning_model"] is not None
    assert len(forecast["combined_forecast"]) > 0

    # 4. Inventory API
    print("\n4. Testing Inventory API...")
    inv_plan = main.get_inventory_get("SKU_001", 150.0)
    print("Inventory Plan for SKU_001 (Stock=150.0):")
    print("  - product_id:", inv_plan["product_id"])
    print("  - eoq:", inv_plan["eoq"])
    print("  - safety_stock:", inv_plan["safety_stock"])
    print("  - reorder_point:", inv_plan["reorder_point"])
    print("  - recommended_order_qty:", inv_plan["recommended_order_qty"])
    print("  - days_remaining:", inv_plan["days_remaining"])
    print("  - inventory_health:", inv_plan["inventory_health"])
    print("  - stockout_probability:", inv_plan["stockout_probability"])

    assert "eoq" in inv_plan
    assert "safety_stock" in inv_plan
    assert "reorder_point" in inv_plan
    assert "recommended_order_qty" in inv_plan
    assert "days_remaining" in inv_plan
    assert "inventory_health" in inv_plan
    assert "stockout_probability" in inv_plan

    # 5. Inventory Risk API
    print("\n5. Testing Inventory Risk API...")
    risk_all = main.get_all_risk()
    print("Total inventory risk records returned:", len(risk_all["inventory_risk"]))
    sample_risk = risk_all["inventory_risk"][0]
    print("Sample Risk Record:")
    print("  - product_id:", sample_risk["product_id"])
    print("  - store_id:", sample_risk["store_id"])
    print("  - category:", sample_risk["category"])
    print("  - risk_score:", sample_risk["risk_score"])
    print("  - risk_level:", sample_risk["risk_level"])
    print("  - confidence:", sample_risk["confidence"])
    print("  - suggested_action:", sample_risk["suggested_action"][:60] + "...")
    print("  - top_risk_factors count:", len(sample_risk["top_risk_factors"]))

    assert "risk_score" in sample_risk
    assert "risk_level" in sample_risk
    assert "confidence" in sample_risk
    assert "suggested_action" in sample_risk
    assert "top_risk_factors" in sample_risk
    assert "product_id" in sample_risk
    assert "store_id" in sample_risk
    assert "category" in sample_risk

    risk_single = main.get_single_risk("SKU_001")
    print(f"Single risk query for SKU_001 returned {len(risk_single['risk_records'])} store records.")
    assert len(risk_single["risk_records"]) > 0

    print("\n" + "=" * 60)
    print("ALL API CONTRACTS VERIFIED AND FUNCTIONING PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_backend_functions()
