/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.jni;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Configuration manager for FusedTransformAggregate operations.
 * 
 * This class provides a centralized way to configure the fused transform + aggregate
 * functionality in Spark Rapids. Configuration can be set via:
 * <ul>
 *   <li>System properties (spark.rapids.sql.fused.*)</li>
 *   <li>Programmatic API</li>
 * </ul>
 * 
 * <h2>Spark Configuration Properties</h2>
 * <table>
 *   <tr>
 *     <th>Property</th>
 *     <th>Default</th>
 *     <th>Description</th>
 *   </tr>
 *   <tr>
 *     <td>spark.rapids.sql.fused.transform.aggregate.enabled</td>
 *     <td>true</td>
 *     <td>Enable/disable fused transform + aggregate operations</td>
 *   </tr>
 *   <tr>
 *     <td>spark.rapids.sql.fused.transform.aggregate.mode</td>
 *     <td>AUTO</td>
 *     <td>Execution mode: AUTO, HAND_WRITTEN_KERNEL, JIT_TRANSFORM, FUSED_1PASS</td>
 *   </tr>
 *   <tr>
 *     <td>spark.rapids.sql.fused.transform.aggregate.1pass.groupThreshold</td>
 *     <td>1000000</td>
 *     <td>Max groups for fused 1-pass mode (L2 cache optimization)</td>
 *   </tr>
 *   <tr>
 *     <td>spark.rapids.sql.fused.transform.aggregate.warpReduction</td>
 *     <td>true</td>
 *     <td>Enable warp-level reduction optimization</td>
 *   </tr>
 *   <tr>
 *     <td>spark.rapids.sql.fused.transform.aggregate.perfectHash</td>
 *     <td>true</td>
 *     <td>Enable perfect hash optimization for small group counts</td>
 *   </tr>
 * </table>
 * 
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Via Spark configuration
 * spark.conf().set("spark.rapids.sql.fused.transform.aggregate.enabled", "true");
 * spark.conf().set("spark.rapids.sql.fused.transform.aggregate.mode", "FUSED_1PASS");
 * 
 * // Via programmatic API
 * FusedTransformAggregateConfig.setEnabled(true);
 * FusedTransformAggregateConfig.setExecutionMode(ExecutionMode.FUSED_1PASS);
 * }</pre>
 */
public class FusedTransformAggregateConfig {
    private static final Logger LOG = LoggerFactory.getLogger(FusedTransformAggregateConfig.class);
    
    // =========================================================================
    // Configuration Property Names
    // =========================================================================
    
    /** Base prefix for all fused transform aggregate properties */
    public static final String CONF_PREFIX = "spark.rapids.sql.fused.transform.aggregate.";
    
    /** Enable/disable fused transform + aggregate */
    public static final String CONF_ENABLED = CONF_PREFIX + "enabled";
    
    /** Execution mode */
    public static final String CONF_MODE = CONF_PREFIX + "mode";
    
    /** Group threshold for fused 1-pass mode */
    public static final String CONF_1PASS_GROUP_THRESHOLD = CONF_PREFIX + "1pass.groupThreshold";
    
    /** Enable warp reduction */
    public static final String CONF_WARP_REDUCTION = CONF_PREFIX + "warpReduction";
    
    /** Enable perfect hash */
    public static final String CONF_PERFECT_HASH = CONF_PREFIX + "perfectHash";
    
    /** Minimum expressions to trigger fusion */
    public static final String CONF_MIN_EXPRESSIONS = CONF_PREFIX + "minExpressions";
    
    // =========================================================================
    // Default Values
    // =========================================================================
    
    public static final boolean DEFAULT_ENABLED = true;
    public static final String DEFAULT_MODE = "AUTO";
    public static final int DEFAULT_1PASS_GROUP_THRESHOLD = 1_000_000;
    public static final boolean DEFAULT_WARP_REDUCTION = true;
    public static final boolean DEFAULT_PERFECT_HASH = true;
    public static final int DEFAULT_MIN_EXPRESSIONS = 1;
    
    // =========================================================================
    // Instance State (Thread-safe via volatile)
    // =========================================================================
    
    private static volatile boolean enabled = DEFAULT_ENABLED;
    private static volatile FusedTransformAggregate.ExecutionMode executionMode = 
        FusedTransformAggregate.ExecutionMode.AUTO;
    private static volatile int fused1PassGroupThreshold = DEFAULT_1PASS_GROUP_THRESHOLD;
    private static volatile boolean warpReductionEnabled = DEFAULT_WARP_REDUCTION;
    private static volatile boolean perfectHashEnabled = DEFAULT_PERFECT_HASH;
    private static volatile int minExpressions = DEFAULT_MIN_EXPRESSIONS;
    
    // Prevent instantiation
    private FusedTransformAggregateConfig() {}
    
    // =========================================================================
    // Static Initialization from System Properties
    // =========================================================================
    
    static {
        loadFromSystemProperties();
    }
    
    /**
     * Load configuration from system properties.
     * Called automatically on class load.
     */
    public static void loadFromSystemProperties() {
        // Load enabled
        String enabledStr = System.getProperty(CONF_ENABLED);
        if (enabledStr != null) {
            enabled = Boolean.parseBoolean(enabledStr);
            LOG.info("Loaded {} = {}", CONF_ENABLED, enabled);
        }
        
        // Load execution mode
        String modeStr = System.getProperty(CONF_MODE);
        if (modeStr != null) {
            try {
                executionMode = FusedTransformAggregate.ExecutionMode.valueOf(modeStr.toUpperCase());
                LOG.info("Loaded {} = {}", CONF_MODE, executionMode);
            } catch (IllegalArgumentException e) {
                LOG.warn("Invalid execution mode '{}', using default: {}", modeStr, DEFAULT_MODE);
            }
        }
        
        // Load 1-pass group threshold
        String thresholdStr = System.getProperty(CONF_1PASS_GROUP_THRESHOLD);
        if (thresholdStr != null) {
            try {
                fused1PassGroupThreshold = Integer.parseInt(thresholdStr);
                LOG.info("Loaded {} = {}", CONF_1PASS_GROUP_THRESHOLD, fused1PassGroupThreshold);
            } catch (NumberFormatException e) {
                LOG.warn("Invalid group threshold '{}', using default: {}", 
                    thresholdStr, DEFAULT_1PASS_GROUP_THRESHOLD);
            }
        }
        
        // Load warp reduction
        String warpStr = System.getProperty(CONF_WARP_REDUCTION);
        if (warpStr != null) {
            warpReductionEnabled = Boolean.parseBoolean(warpStr);
            LOG.info("Loaded {} = {}", CONF_WARP_REDUCTION, warpReductionEnabled);
        }
        
        // Load perfect hash
        String hashStr = System.getProperty(CONF_PERFECT_HASH);
        if (hashStr != null) {
            perfectHashEnabled = Boolean.parseBoolean(hashStr);
            LOG.info("Loaded {} = {}", CONF_PERFECT_HASH, perfectHashEnabled);
        }
        
        // Load min expressions
        String minExprStr = System.getProperty(CONF_MIN_EXPRESSIONS);
        if (minExprStr != null) {
            try {
                minExpressions = Integer.parseInt(minExprStr);
                LOG.info("Loaded {} = {}", CONF_MIN_EXPRESSIONS, minExpressions);
            } catch (NumberFormatException e) {
                LOG.warn("Invalid min expressions '{}', using default: {}", 
                    minExprStr, DEFAULT_MIN_EXPRESSIONS);
            }
        }
    }
    
    // =========================================================================
    // Getters
    // =========================================================================
    
    /**
     * Check if fused transform + aggregate is enabled.
     */
    public static boolean isEnabled() {
        return enabled;
    }
    
    /**
     * Get the current execution mode.
     */
    public static FusedTransformAggregate.ExecutionMode getExecutionMode() {
        return executionMode;
    }
    
    /**
     * Get the group threshold for fused 1-pass mode.
     */
    public static int getFused1PassGroupThreshold() {
        return fused1PassGroupThreshold;
    }
    
    /**
     * Check if warp reduction is enabled.
     */
    public static boolean isWarpReductionEnabled() {
        return warpReductionEnabled;
    }
    
    /**
     * Check if perfect hash is enabled.
     */
    public static boolean isPerfectHashEnabled() {
        return perfectHashEnabled;
    }
    
    /**
     * Get the minimum number of expressions to trigger fusion.
     */
    public static int getMinExpressions() {
        return minExpressions;
    }
    
    // =========================================================================
    // Setters
    // =========================================================================
    
    /**
     * Enable or disable fused transform + aggregate.
     */
    public static void setEnabled(boolean value) {
        enabled = value;
        LOG.info("Set {} = {}", CONF_ENABLED, value);
    }
    
    /**
     * Set the execution mode.
     */
    public static void setExecutionMode(FusedTransformAggregate.ExecutionMode mode) {
        executionMode = mode;
        LOG.info("Set {} = {}", CONF_MODE, mode);
    }
    
    /**
     * Set the group threshold for fused 1-pass mode.
     */
    public static void setFused1PassGroupThreshold(int threshold) {
        fused1PassGroupThreshold = threshold;
        LOG.info("Set {} = {}", CONF_1PASS_GROUP_THRESHOLD, threshold);
    }
    
    /**
     * Enable or disable warp reduction.
     */
    public static void setWarpReductionEnabled(boolean value) {
        warpReductionEnabled = value;
        LOG.info("Set {} = {}", CONF_WARP_REDUCTION, value);
    }
    
    /**
     * Enable or disable perfect hash.
     */
    public static void setPerfectHashEnabled(boolean value) {
        perfectHashEnabled = value;
        LOG.info("Set {} = {}", CONF_PERFECT_HASH, value);
    }
    
    /**
     * Set the minimum number of expressions to trigger fusion.
     */
    public static void setMinExpressions(int value) {
        minExpressions = value;
        LOG.info("Set {} = {}", CONF_MIN_EXPRESSIONS, value);
    }
    
    // =========================================================================
    // Configuration Builder
    // =========================================================================
    
    /**
     * Create a FusedTransformAggregate.Config from current settings.
     * 
     * @return Config object with current settings
     */
    public static FusedTransformAggregate.Config buildConfig() {
        return new FusedTransformAggregate.Config()
            .setExecutionMode(executionMode)
            .setEnableWarpReduction(warpReductionEnabled)
            .setEnablePerfectHash(perfectHashEnabled)
            .setFused1PassGroupThreshold(fused1PassGroupThreshold);
    }
    
    /**
     * Check if fusion should be used based on current settings and expression count.
     * 
     * @param numExpressions Number of expressions to fuse
     * @param numGroupByCols Number of group-by columns
     * @return true if fusion should be used
     */
    public static boolean shouldUseFusion(int numExpressions, int numGroupByCols) {
        if (!enabled) {
            return false;
        }
        if (numExpressions < minExpressions) {
            return false;
        }
        return FusedTransformAggregate.shouldUseFused(numExpressions, numGroupByCols);
    }
    
    // =========================================================================
    // Reset to Defaults
    // =========================================================================
    
    /**
     * Reset all configuration to default values.
     */
    public static void resetToDefaults() {
        enabled = DEFAULT_ENABLED;
        executionMode = FusedTransformAggregate.ExecutionMode.AUTO;
        fused1PassGroupThreshold = DEFAULT_1PASS_GROUP_THRESHOLD;
        warpReductionEnabled = DEFAULT_WARP_REDUCTION;
        perfectHashEnabled = DEFAULT_PERFECT_HASH;
        minExpressions = DEFAULT_MIN_EXPRESSIONS;
        LOG.info("Reset fused transform aggregate configuration to defaults");
    }
    
    // =========================================================================
    // Debug / Info
    // =========================================================================
    
    /**
     * Get a summary of current configuration for logging/debugging.
     */
    public static String getConfigSummary() {
        return String.format(
            "FusedTransformAggregateConfig{enabled=%s, mode=%s, " +
            "1passGroupThreshold=%d, warpReduction=%s, perfectHash=%s, minExpressions=%d}",
            enabled, executionMode, fused1PassGroupThreshold,
            warpReductionEnabled, perfectHashEnabled, minExpressions);
    }
    
    /**
     * Log current configuration at INFO level.
     */
    public static void logConfiguration() {
        LOG.info(getConfigSummary());
    }
}


