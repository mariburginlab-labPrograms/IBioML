#!/usr/bin/env python3
"""Test para mostrar diferentes configuraciones de capas exclusivas en modelos dual."""

import torch
from ibioml.models import DualOutputMLPModel, DualOutputLSTMModel

def print_model_config(title, model):
    """Imprimir configuración y estructura del modelo."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total parámetros: {total_params:,}")
    
    # Desglose por componente
    print("\n📋 Desglose por componente:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  - {name}: {params:,} parámetros")
    print()


def test_mlp_configurations():
    """Probar diferentes configuraciones del DualOutputMLPModel."""
    print("\n" + "#"*80)
    print("# DualOutputMLPModel - Diferentes Configuraciones")
    print("#"*80)
    
    input_size = 50
    hidden_size = 128
    output_size = 1
    num_layers = 3
    dropout = 0.2
    
    # Configuración 1: CON capas exclusivas (default)
    model1 = DualOutputMLPModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout,
        use_exclusive=True  # Default
    )
    print_model_config("Config 1: CON capas exclusivas (1 capa, mismo tamaño)", model1)
    
    # Configuración 2: SIN capas exclusivas
    model2 = DualOutputMLPModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout,
        use_exclusive=False
    )
    print_model_config("Config 2: SIN capas exclusivas", model2)
    
    # Configuración 3: Con capas exclusivas más pequeñas
    model3 = DualOutputMLPModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout,
        use_exclusive=True,
        exclusive_hidden_size=64  # Más pequeño que hidden_size
    )
    print_model_config("Config 3: Capas exclusivas más pequeñas (64 vs 128)", model3)
    
    # Configuración 4: Con capas exclusivas más grandes
    model4 = DualOutputMLPModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout,
        use_exclusive=True,
        exclusive_hidden_size=256  # Más grande que hidden_size
    )
    print_model_config("Config 4: Capas exclusivas más grandes (256 vs 128)", model4)
    
    # Configuración 5: Con múltiples capas exclusivas
    model5 = DualOutputMLPModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout,
        use_exclusive=True,
        num_exclusive_layers=3  # 3 capas exclusivas por output
    )
    print_model_config("Config 5: Múltiples capas exclusivas (3 capas)", model5)
    
    # Configuración 6: Combinación personalizada
    model6 = DualOutputMLPModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout,
        use_exclusive=True,
        exclusive_hidden_size=96,
        num_exclusive_layers=2
    )
    print_model_config("Config 6: Custom (2 capas de 96 unidades)", model6)
    
    # Test forward pass
    print("="*80)
    print("TEST FORWARD PASS")
    print("="*80)
    
    x = torch.randn(16, input_size)
    models = [model1, model2, model3, model4, model5, model6]
    
    for i, model in enumerate(models, 1):
        out1, out2 = model(x.float())
        print(f"Config {i}: Input {x.shape} -> Output1 {out1.shape}, Output2 {out2.shape} ✓")
    
    print("\n✅ Todos los forward passes exitosos!")


def test_rnn_configurations():
    """Probar diferentes configuraciones del DualOutputLSTMModel."""
    print("\n\n" + "#"*80)
    print("# DualOutputLSTMModel - Diferentes Configuraciones")
    print("#"*80)
    
    input_size = 50
    hidden_size = 128
    output_size = 1
    num_layers = 2
    dropout = 0.2
    
    # Configuración 1: CON capas exclusivas (default)
    model1 = DualOutputLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout,
        use_exclusive=True
    )
    print_model_config("LSTM Config 1: CON capas exclusivas", model1)
    
    # Configuración 2: SIN capas exclusivas
    model2 = DualOutputLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout,
        use_exclusive=False
    )
    print_model_config("LSTM Config 2: SIN capas exclusivas", model2)
    
    # Configuración 3: Con múltiples capas exclusivas
    model3 = DualOutputLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout,
        use_exclusive=True,
        exclusive_hidden_size=96,
        num_exclusive_layers=2
    )
    print_model_config("LSTM Config 3: 2 capas exclusivas de 96 unidades", model3)
    
    # Test forward pass
    print("="*80)
    print("TEST FORWARD PASS (RNN)")
    print("="*80)
    
    x = torch.randn(16, 10, input_size)  # (batch, seq_len, features)
    models = [model1, model2, model3]
    
    for i, model in enumerate(models, 1):
        out1, out2 = model(x.float())
        print(f"LSTM Config {i}: Input {x.shape} -> Output1 {out1.shape}, Output2 {out2.shape} ✓")
    
    print("\n✅ Todos los forward passes RNN exitosos!")


def comparison_table():
    """Crear una tabla comparativa de parámetros."""
    print("\n\n" + "="*80)
    print("TABLA COMPARATIVA DE CONFIGURACIONES")
    print("="*80)
    
    configs = [
        ("Sin exclusive", False, None, 1),
        ("Con exclusive (1x128)", True, 128, 1),
        ("Con exclusive (1x64)", True, 64, 1),
        ("Con exclusive (1x256)", True, 256, 1),
        ("Con exclusive (2x128)", True, 128, 2),
        ("Con exclusive (3x128)", True, 128, 3),
    ]
    
    print("\n{:<30} {:>15} {:>20}".format("Configuración", "Total Params", "Exclusive Params"))
    print("-" * 80)
    
    for name, use_exc, exc_size, num_exc in configs:
        model = DualOutputMLPModel(
            input_size=50,
            hidden_size=128,
            output_size=1,
            num_layers=3,
            dropout=0.2,
            use_exclusive=use_exc,
            exclusive_hidden_size=exc_size,
            num_exclusive_layers=num_exc
        )
        
        total = sum(p.numel() for p in model.parameters())
        
        if use_exc:
            exc1 = sum(p.numel() for p in model.exclusive_head1.parameters())
            exc2 = sum(p.numel() for p in model.exclusive_head2.parameters())
            exc_total = exc1 + exc2
        else:
            exc_total = 0
        
        print("{:<30} {:>15,} {:>20,}".format(name, total, exc_total))
    
    print()


if __name__ == "__main__":
    print("\n" + "*"*80)
    print("*" + " "*78 + "*")
    print("*" + " "*20 + "TEST DE CAPAS EXCLUSIVAS PARAMETRIZADAS" + " "*19 + "*")
    print("*" + " "*78 + "*")
    print("*"*80)
    
    try:
        test_mlp_configurations()
        test_rnn_configurations()
        comparison_table()
        
        print("\n" + "="*80)
        print("✅ TODOS LOS TESTS COMPLETADOS EXITOSAMENTE")
        print("="*80 + "\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ ERROR: {e}")
        print("="*80 + "\n")
        import traceback
        traceback.print_exc()
        raise
