import streamlit as st
import pandas as pd
import os
import subprocess
from datetime import datetime
from langchain_experimental.agents import create_csv_agent
from langchain_community.llms import Ollama
import tempfile
import time
import requests
from langchain.agents import initialize_agent, AgentType


from langchain.tools import Tool

from langchain.tools import Tool
import pandas as pd
import streamlit as st

import pandas as pd
import numpy as np
from langchain.tools import Tool
import streamlit as st

class SalesTools:
    def __init__(self, df):
        self.df = df.copy()
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Handle data preprocessing and type conversions"""
        # Date conversion
        date_cols = ['order_date', 'ship_date', 'delivery_date']
        for col in date_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                except Exception:
                    pass
        
        # Numeric conversion
        numeric_cols = ['sales', 'quantity', 'price', 'cost', 'profit']
        for col in numeric_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except Exception:
                    pass
    
    # ======================================================
    # Core Tool Functions (with proper signatures)
    # ======================================================
    
    def total_sales_tool(self, _: str) -> str:
        return self._total_sales()
    
    def total_orders_tool(self, _: str) -> str:
        return self._total_orders()
    
    def top_products_tool(self, _: str) -> str:
        return self._top_products()
    
    def worst_products_tool(self, _: str) -> str:
        return self._worst_products()
    
    def sales_trend_tool(self, _: str) -> str:
        return self._sales_trend()
    
    def top_customers_tool(self, _: str) -> str:
        return self._top_customers()
    
    def customer_acquisition_tool(self, _: str) -> str:
        return self._customer_acquisition()
    
    def regional_sales_tool(self, _: str) -> str:
        return self._regional_sales()
    
    def channel_performance_tool(self, _: str) -> str:
        return self._channel_performance()
    
    def inventory_turnover_tool(self, _: str) -> str:
        return self._inventory_turnover()
    
    def product_seasonality_tool(self, _: str) -> str:
        return self._product_seasonality()
    
    def customer_lifetime_value_tool(self, _: str) -> str:
        return self._customer_lifetime_value()
    
    def return_rate_tool(self, _: str) -> str:
        return self._return_rate()
    
    def shipping_performance_tool(self, _: str) -> str:
        return self._shipping_performance()
    
    def discount_impact_tool(self, _: str) -> str:
        return self._discount_impact()
    
    def basket_analysis_tool(self, _: str) -> str:
        return self._basket_analysis()
    
    def cohort_analysis_tool(self, _: str) -> str:
        return self._cohort_analysis()
    
    def sales_forecast_tool(self, _: str) -> str:
        return self._sales_forecast()
    
    def product_recommendation_tool(self, input_str: str) -> str:
        return self._product_recommendation(input_str)
    
    # ======================================================
    # Implementation Methods
    # ======================================================
    
    def _total_sales(self) -> str:
        try:
            if "sales" not in self.df.columns:
                return "Error: 'sales' column not found"
            total = self.df["sales"].sum()
            return f"Total sales: ${total:,.2f}"
        except Exception as e:
            return f"Error calculating total sales: {str(e)}"
    
    def _total_orders(self) -> str:
        try:
            if "order_id" in self.df.columns:
                count = self.df["order_id"].nunique()
                return f"Total orders: {count}"
            elif "transaction_id" in self.df.columns:
                count = self.df["transaction_id"].nunique()
                return f"Total orders: {count}"
            else:
                return "Error: No order identifier column found"
        except Exception as e:
            return f"Error counting orders: {str(e)}"
    
    def _top_products(self, n: int = 5) -> str:
        try:
            if "product" not in self.df.columns or "sales" not in self.df.columns:
                return "Error: Missing required columns"
                
            top = (
                self.df.groupby("product")["sales"]
                .sum()
                .sort_values(ascending=False)
                .head(n)
            )
            return f"Top {n} products by sales:\n{top.to_string()}"
        except Exception as e:
            return f"Error finding top products: {str(e)}"
    
    def _worst_products(self, n: int = 5) -> str:
        try:
            if "product" not in self.df.columns or "sales" not in self.df.columns:
                return "Error: Missing required columns"
                
            worst = (
                self.df.groupby("product")["sales"]
                .sum()
                .sort_values()
                .head(n)
            )
            return f"Worst {n} products by sales:\n{worst.to_string()}"
        except Exception as e:
            return f"Error finding worst products: {str(e)}"
    
    def _sales_trend(self, period: str = "M") -> str:
        try:
            if "order_date" not in self.df.columns or "sales" not in self.df.columns:
                return "Error: Missing required columns"
                
            if not pd.api.types.is_datetime64_any_dtype(self.df["order_date"]):
                return "Error: 'order_date' is not datetime"
                
            df = self.df.copy()
            df.set_index("order_date", inplace=True)
            resampled = df["sales"].resample(period).sum()
            
            # Calculate growth
            growth = (resampled.pct_change() * 100).fillna(0)
            
            result = f"Sales trend ({period}):\n{resampled.to_string()}"
            result += f"\n\nGrowth rate (%):\n{growth.to_string()}"
            return result
        except Exception as e:
            return f"Error calculating sales trend: {str(e)}"
    
    def _top_customers(self, n: int = 5) -> str:
        try:
            if "customer_id" not in self.df.columns or "sales" not in self.df.columns:
                return "Error: Missing required columns"
                
            top = (
                self.df.groupby("customer_id")["sales"]
                .sum()
                .sort_values(ascending=False)
                .head(n)
            )
            return f"Top {n} customers by spend:\n{top.to_string()}"
        except Exception as e:
            return f"Error finding top customers: {str(e)}"
    
    def _customer_acquisition(self) -> str:
        try:
            if "customer_id" not in self.df.columns or "order_date" not in self.df.columns:
                return "Error: Missing required columns"
                
            # Find first purchase date for each customer
            first_purchase = self.df.groupby("customer_id")["order_date"].min()
            
            # Count new customers per month
            monthly_acquisition = first_purchase.dt.to_period("M").value_counts().sort_index()
            
            return f"New customers by month:\n{monthly_acquisition.to_string()}"
        except Exception as e:
            return f"Error calculating customer acquisition: {str(e)}"
    
    def _regional_sales(self) -> str:
        try:
            if "region" not in self.df.columns or "sales" not in self.df.columns:
                return "Error: Missing required columns"
                
            regional = self.df.groupby("region")["sales"].sum().sort_values(ascending=False)
            return f"Sales by region:\n{regional.to_string()}"
        except Exception as e:
            return f"Error calculating regional sales: {str(e)}"
    
    def _channel_performance(self) -> str:
        try:
            if "channel" not in self.df.columns or "sales" not in self.df.columns:
                return "Error: Missing required columns"
                
            channel = self.df.groupby("channel")["sales"].sum().sort_values(ascending=False)
            return f"Sales by channel:\n{channel.to_string()}"
        except Exception as e:
            return f"Error calculating channel performance: {str(e)}"
    
    def _inventory_turnover(self) -> str:
        try:
            if "product" not in self.df.columns or "quantity" not in self.df.columns:
                return "Error: Missing required columns"
                
            turnover = (
                self.df.groupby("product")["quantity"]
                .sum()
                .sort_values(ascending=False)
            )
            return f"Inventory turnover by product:\n{turnover.to_string()}"
        except Exception as e:
            return f"Error calculating inventory turnover: {str(e)}"
    
    def _product_seasonality(self) -> str:
        try:
            if "product" not in self.df.columns or "order_date" not in self.df.columns or "quantity" not in self.df.columns:
                return "Error: Missing required columns"
                
            df = self.df.copy()
            df["month"] = df["order_date"].dt.month
            seasonality = df.groupby(["product", "month"])["quantity"].sum().unstack()
            return f"Product seasonality (monthly sales):\n{seasonality.to_string()}"
        except Exception as e:
            return f"Error calculating product seasonality: {str(e)}"
    
    def _customer_lifetime_value(self) -> str:
        try:
            if "customer_id" not in self.df.columns or "sales" not in self.df.columns:
                return "Error: Missing required columns"
                
            clv = self.df.groupby("customer_id")["sales"].sum().sort_values(ascending=False)
            return f"Customer Lifetime Value (CLV):\n{clv.to_string()}"
        except Exception as e:
            return f"Error calculating CLV: {str(e)}"
    
    def _return_rate(self) -> str:
        try:
            if "return_status" not in self.df.columns:
                return "Error: 'return_status' column not found"
                
            total_orders = self.df["order_id"].nunique()
            return_count = self.df[self.df["return_status"] == "Returned"]["order_id"].nunique()
            rate = (return_count / total_orders) * 100 if total_orders > 0 else 0
            return f"Return rate: {rate:.2f}% ({return_count} returns out of {total_orders} orders)"
        except Exception as e:
            return f"Error calculating return rate: {str(e)}"
    
    def _shipping_performance(self) -> str:
        try:
            if "order_date" not in self.df.columns or "ship_date" not in self.df.columns:
                return "Error: Missing required date columns"
                
            df = self.df.copy()
            df["processing_time"] = (df["ship_date"] - df["order_date"]).dt.days
            avg_time = df["processing_time"].mean()
            return f"Average order processing time: {avg_time:.1f} days"
        except Exception as e:
            return f"Error calculating shipping performance: {str(e)}"
    
    def _discount_impact(self) -> str:
        try:
            if "discount" not in self.df.columns or "quantity" not in self.df.columns:
                return "Error: Missing discount or quantity data"
                
            correlation = self.df[["discount", "quantity"]].corr().iloc[0,1]
            return f"Discount-quantity correlation: {correlation:.2f} (values near 1 indicate discounts boost sales)"
        except Exception as e:
            return f"Error analyzing discount impact: {str(e)}"
    
    def _basket_analysis(self) -> str:
        try:
            if "order_id" not in self.df.columns or "product" not in self.df.columns:
                return "Error: Missing required columns"
                
            # Find frequently bought together products
            from mlxtend.frequent_patterns import apriori, association_rules
            
            basket = (self.df.groupby(['order_id', 'product'])['quantity']
                      .sum().unstack().reset_index().fillna(0))
            basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
            
            frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
            
            # Format top 5 rules
            top_rules = rules.sort_values('confidence', ascending=False).head(5)
            result = "Top product associations (frequently bought together):\n"
            for _, row in top_rules.iterrows():
                result += f"- {', '.join(row['antecedents'])} => {', '.join(row['consequents'])} (confidence: {row['confidence']:.2f})\n"
            
            return result
        except Exception as e:
            return f"Error performing basket analysis: {str(e)}"
    
    def _cohort_analysis(self) -> str:
        try:
            if "customer_id" not in self.df.columns or "order_date" not in self.df.columns:
                return "Error: Missing required columns"
                
            # Create acquisition cohorts
            df = self.df.copy()
            df["acquisition_month"] = df.groupby("customer_id")["order_date"].transform("min").dt.to_period("M")
            df["order_month"] = df["order_date"].dt.to_period("M")
            
            # Calculate cohort index
            df["cohort_index"] = (df["order_month"] - df["acquisition_month"]).apply(
                lambda x: x.n if x.n >= 0 else 0
            )
            
            # Retention analysis
            cohort_data = df.groupby(["acquisition_month", "cohort_index"])["customer_id"].nunique().unstack()
            retention = cohort_data.divide(cohort_data.iloc[:, 0], axis=0)
            
            return f"Cohort retention rates:\n{retention.to_string()}"
        except Exception as e:
            return f"Error performing cohort analysis: {str(e)}"
    
    def _sales_forecast(self) -> str:
        try:
            if "order_date" not in self.df.columns or "sales" not in self.df.columns:
                return "Error: Missing required columns"
                
            # Simple moving average forecast
            df = self.df.set_index("order_date").resample("D")["sales"].sum().reset_index()
            df["7day_ma"] = df["sales"].rolling(window=7).mean()
            forecast = df[["order_date", "7day_ma"]].tail(30)
            
            return f"30-day sales forecast (7-day moving average):\n{forecast.to_string()}"
        except Exception as e:
            return f"Error generating sales forecast: {str(e)}"
    
    def _product_recommendation(self, customer_id: str) -> str:
        try:
            if "customer_id" not in self.df.columns or "product" not in self.df.columns:
                return "Error: Missing required columns"
                
            # Simple collaborative filtering
            customer_history = self.df[self.df["customer_id"] == customer_id]["product"].unique()
            similar_customers = self.df[self.df["product"].isin(customer_history)]
            recommendations = (
                similar_customers[~similar_customers["product"].isin(customer_history)]
                .groupby("product")["sales"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )
            
            if recommendations.empty:
                return f"No personalized recommendations found for customer {customer_id}"
                
            return f"Top recommendations for {customer_id}:\n{recommendations.to_string()}"
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"
    
    # ======================================================
    # Tool List Generator
    # ======================================================
    
    def get_tool_list(self):
        """Create LangChain tools with proper binding"""
        return [
            # Core metrics
            Tool.from_function(
                func=self.total_sales_tool,
                name="Total_Sales",
                description="Calculates total sales revenue"
            ),
            Tool.from_function(
                func=self.total_orders_tool,
                name="Total_Orders",
                description="Counts total number of orders"
            ),
            
            # Product analysis
            Tool.from_function(
                func=self.top_products_tool,
                name="Top_Products",
                description="Finds top selling products by revenue"
            ),
            Tool.from_function(
                func=self.worst_products_tool,
                name="Worst_Products",
                description="Finds worst performing products by revenue"
            ),
            Tool.from_function(
                func=self.product_seasonality_tool,
                name="Product_Seasonality",
                description="Analyzes seasonal patterns for products"
            ),
            Tool.from_function(
                func=self.inventory_turnover_tool,
                name="Inventory_Turnover",
                description="Calculates inventory turnover rate by product"
            ),
            
            # Customer analysis
            Tool.from_function(
                func=self.top_customers_tool,
                name="Top_Customers",
                description="Identifies highest spending customers"
            ),
            Tool.from_function(
                func=self.customer_acquisition_tool,
                name="Customer_Acquisition",
                description="Analyzes new customer acquisition trends"
            ),
            Tool.from_function(
                func=self.customer_lifetime_value_tool,
                name="Customer_Lifetime_Value",
                description="Calculates customer lifetime value (CLV)"
            ),
            Tool.from_function(
                func=self.cohort_analysis_tool,
                name="Cohort_Analysis",
                description="Performs cohort analysis of customer retention"
            ),
            
            # Sales analysis
            Tool.from_function(
                func=self.sales_trend_tool,
                name="Sales_Trend",
                description="Analyzes sales trends over time"
            ),
            Tool.from_function(
                func=self.regional_sales_tool,
                name="Regional_Sales",
                description="Compares sales performance by region"
            ),
            Tool.from_function(
                func=self.channel_performance_tool,
                name="Channel_Performance",
                description="Evaluates sales performance by channel"
            ),
            Tool.from_function(
                func=self.sales_forecast_tool,
                name="Sales_Forecast",
                description="Generates sales forecast based on historical data"
            ),
            
            # Operational metrics
            Tool.from_function(
                func=self.return_rate_tool,
                name="Return_Rate",
                description="Calculates product return rate"
            ),
            Tool.from_function(
                func=self.shipping_performance_tool,
                name="Shipping_Performance",
                description="Measures order fulfillment speed"
            ),
            Tool.from_function(
                func=self.discount_impact_tool,
                name="Discount_Impact",
                description="Analyzes impact of discounts on sales volume"
            ),
            
            # Advanced analytics
            Tool.from_function(
                func=self.basket_analysis_tool,
                name="Basket_Analysis",
                description="Identifies products frequently bought together"
            ),
            Tool.from_function(
                func=self.product_recommendation_tool,
                name="Product_Recommendation",
                description="Generates personalized product recommendations. Input should be a customer ID."
            )
        ]
class DashboardChatbot:
    def __init__(self, df: pd.DataFrame, data_processor: object, predictor: object):
        self.df = df
        self.processor = data_processor
        self.predictor = predictor
        self.chat_history = []
        self.agent = None
        self.llm_ready = False
        self.csv_path = None
        self.initialize_ollama()
        
    def initialize_ollama(self):
        """Check Ollama status without starting it"""
        try:
            # Check if Ollama is responsive
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if any("llama3" in model["name"] for model in models):
                    self.llm_ready = True
                    self._create_agent()
                else:
                    st.error("llama3 model not installed. Run 'ollama pull llama3'")
            else:
                st.error(f"Ollama not responding (status {response.status_code})")
        except requests.exceptions.ConnectionError:
            st.error("Ollama not running. Please start it with 'ollama serve'")
        except Exception as e:
            st.error(f"Ollama check failed: {str(e)}")


    def _create_agent(self):
        """Create the agent with comprehensive validation and error handling"""
        if not self.llm_ready:
            st.warning("⚠️ LLM not ready - agent creation skipped")
            return
            
        if self.df.empty:
            st.error("📊 No data available - agent creation skipped")
            return

        try:
            # Validate required columns
            required_cols = {"sales", "product", "order_date"}
            missing = required_cols - set(self.df.columns)
            if missing:
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
                return
                
            # Check data types
            if not pd.api.types.is_numeric_dtype(self.df["sales"]):
                st.warning("⚠️ 'sales' column is not numeric - attempting conversion")
                try:
                    self.df["sales"] = pd.to_numeric(self.df["sales"], errors="coerce")
                except Exception:
                    st.error("❌ Failed to convert 'sales' to numeric")
                    return

            # Initialize LLM
            try:
                llm = Ollama(model="llama3", temperature=0.2)
            except Exception as e:
                st.error(f"❌ LLM initialization failed: {str(e)}")
                return

            # Create tools
            try:
                tools = SalesTools(self.df).get_tool_list()
                if not tools:
                    st.error("❌ No tools created")
                    return
                    
                # Verify tool signatures
                for tool in tools:
                    if not callable(tool.func):
                        st.error(f"❌ Tool {tool.name} is not callable")
                        return
                        
            except Exception as e:
                st.error(f"❌ Tool creation failed: {str(e)}")
                return

            # Create agent with enhanced configuration
            try:
                self.agent = initialize_agent(
                    tools=tools,
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=10,
                    early_stopping_method="generate"
                )
                st.success("✅ Agent created successfully!")
            except Exception as e:
                st.error(f"❌ Agent initialization failed: {str(e)}")
                self.agent = None
                
        except Exception as e:
            st.error(f"❌ Unexpected error in agent creation: {str(e)}")
            import traceback
            st.text(traceback.format_exc())  # Show full traceback
            self.agent = None

    def generate_response(self, user_input: str) -> str:
        """Generate response with robust error handling"""
        if not self.llm_ready:
            return "🛠️ AI service unavailable. Please ensure Ollama is running with llama3 model."
            
        if not self.agent:
            return "📊 Data agent not initialized. Try reloading the page."
            
        try:
            # Simplify complex questions
            if "trend" in user_input.lower() or "over time" in user_input.lower():
                user_input = "Show monthly sales totals"
                
            # Run the agent
            response = self.agent.run(user_input)
            return self._post_process(response)
            
        except Exception as e:
            # Handle common errors
            error_msg = str(e)
            if "context length" in error_msg:
                return "⚠️ Query too complex. Try a simpler question."
            elif "memory" in error_msg.lower():
                return "⚠️ Out of memory. Try analyzing a smaller dataset."
            else:
                return f"❌ Error: {error_msg[:200]}..."

    def _post_process(self, response: str) -> str:
        """Enhance raw LLM response"""
        # Clean up response
        response = response.replace("$", "USD ")
        response = response.replace("USD 1000", "USD 1,000")
        response = response.replace("USD 1000000", "USD 1,000,000")
        
        # Add visualization hints
        if any(trigger in response.lower() for trigger in ["trend", "compare", "chart"]):
            response += "\n\n💡 View detailed charts in 'Sales Analytics'"
            
        return response

    def log_interaction(self, user_input: str, response: str):
        """Store conversation history"""
        self.chat_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "ai": response
        })
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]

def render_chat_tab(df: pd.DataFrame, data_processor: object, predictor: object):
    """Main chat interface"""
    st.title("💬 Data Genie AI Assistant")
    
    # Add business context description
    st.markdown("""
    **I'm your sales data assistant!** For best results:
    - Ask specific questions: "Top 5 products by revenue"
    - Use time frames: "Sales last quarter"
    - Compare: "Sales by region"
    """)

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = DashboardChatbot(
            df=df,
            data_processor=data_processor,
            predictor=predictor
        )
        # Add welcome message
        welcome_msg = "Hi! I'm your sales data assistant. Ask me about products, customers, or sales trends!"
        st.session_state.chat_messages.append({"role": "assistant", "content": welcome_msg})
    
    # Display chat history
    for msg in st.session_state.get('chat_messages', []):
        st.chat_message(msg["role"]).write(msg["content"])
    
    # User input
    if prompt := st.chat_input("Ask about sales data..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = st.session_state.chatbot.generate_response(prompt)
                st.write(response)
        
        # Store interaction
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.session_state.chatbot.log_interaction(prompt, response)
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("AI Assistant Status")
        
        if st.session_state.chatbot.llm_ready:
            st.success("✅ Ollama connected")
            if st.session_state.chatbot.agent:
                st.success("✅ Agent ready")
            else:
                st.warning("⚠️ Agent not initialized")
        else:
            st.error("❌ Ollama not available")
            
        if not df.empty:
            st.info(f"📊 {len(df)} rows loaded")
            
        # Ollama management
        st.subheader("Ollama Control")
        if st.button("🔄 Check Ollama Status"):
            st.session_state.chatbot.initialize_ollama()
            st.rerun()
            
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_messages = []
            st.session_state.chatbot.chat_history = []
            st.rerun()
            
        # Data tips
        st.subheader("Tips for Better Results")
        st.markdown("- Keep questions specific")
        st.markdown("- Use time ranges like 'last month'")
        st.markdown("- Example: 'Top 10 products by revenue'")
        
        # System info
        if st.session_state.chatbot.csv_path:
            st.caption(f"Temp CSV: {os.path.basename(st.session_state.chatbot.csv_path)}")