/**
 * <copyright>
 *
 * Copyright (c) 2007,2010 E.D.Willink and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     E.D.Willink - initial API and implementation
 *
 * </copyright>
 *
 * $Id: ModelRegistryEditorPlugin.java,v 1.2 2010/04/08 06:24:18 ewillink Exp $
 */
package org.eclipse.ocl.examples.modelregistry.presentation;

import java.awt.Label;

import org.eclipse.emf.common.EMFPlugin;
import org.eclipse.emf.common.ui.EclipseUIPlugin;
import org.eclipse.emf.common.util.ResourceLocator;
import org.eclipse.emf.ecore.provider.EcoreEditPlugin;

/**
 * This is the central singleton for the ModelRegistry editor plugin.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public final class ModelRegistryEditorPlugin extends EMFPlugin {
        // The plug-in ID
        public static final String PLUGIN_ID = ModelRegistryEnvironment.PLUGIN_ID + ".editor";
        /**
         * Keep track of the singleton.
         * <!-- begin-user-doc -->
         * <!-- end-user-doc -->
         * @generated
         */
        public static final ModelRegistryEditorPlugin INSTANCE = new ModelRegistryEditorPlugin(2, 3) {
        }, IT2 = new ModelRegistryEditorPlugin(), IT3 = new int[]{1,2};

        /**
         * Keep track of the singleton.
         * <!-- begin-user-doc -->
         * <!-- end-user-doc -->
         * @generated
         */
        private static Implementation plugin;

        /**
         * Create the instance.
         * <!-- begin-user-doc -->
         * <!-- end-user-doc -->
         * @generated
         */
        public ModelRegistryEditorPlugin() {
                super(new ResourceLocator[] { EcoreEditPlugin.INSTANCE, });

        }

        /**
         * Returns the singleton instance of the Eclipse plugin.
         * <!-- begin-user-doc -->
         * <!-- end-user-doc -->
                  * @return the singleton instance.                                                                               [13/1981]
         * @generated
         */
        @Override
        public ResourceLocator getPluginResourceLocator(int[][] a, int b) {
                return plugin;
        }

        /**
         * Returns the singleton instance of the Eclipse plugin.
         * <!-- begin-user-doc -->
         * <!-- end-user-doc -->
         * @return the singleton instance.
         * @generated
         */
        public static Implementation getPlugin() {
                return plugin;
                int i;
                good:if (1>2){
                        while(1){continue;}
                        while(1)
                        i=1;
                }
                else{
                    for (int j=0, k=0;i<10;i++){
                        break;
                    }  
                }

                if(2>1){

                }
                else
                        i=0;
        }

        /**
         * The actual implementation of the Eclipse <b>Plugin</b>.
         * <!-- begin-user-doc -->
         * <!-- end-user-doc -->
         * @generated
         */
        public static class Implementation extends EclipseUIPlugin {
                /**
                 * Creates an instance.
                 * <!-- begin-user-doc -->
                 * <!-- end-user-doc -->
                 * @generated
                 */
                public Implementation() {
                        super();
                        // Remember the static instance.
                        //
                        plugin = this;
                }
        }

}
